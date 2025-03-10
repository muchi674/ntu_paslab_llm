"""
Partitioner designer for mixtral-8x7b weights from
https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
"""

from pathlib import Path
import argparse
import glob
import json
import logging

import torch


def ceildiv(a, b):
    # from: https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return -(a // -b)


class Partitioner:

    def __init__(self, model_path: str, design_path: str, output_path: str) -> None:
        self.model_path = Path(model_path)
        self.design_path = Path(design_path)
        self.output_path = Path(output_path) if output_path else self.model_path
        self.model_config, self.expert_map, self.attn_tp_map, self.non_expert_pp_map = (
            self.get_configs()
        )

    def get_configs(self) -> tuple[dict, dict]:

        def get_devices(parallel_size: int, node_id: int):
            if parallel_size is None:
                return []

            devices = []
            for di in range(parallel_size):
                if node_id is None:
                    name = str(di)
                else:
                    name = f"{node_id}-{di}"
                devices.append(name)
            return devices

        with open(self.model_path / "config.json", "r") as model_config_file:
            model_config = json.load(model_config_file)
        with open(self.design_path, "r") as design_file:
            design = json.load(design_file)

        expert_map = {}
        attn_tp_map = {}
        non_expert_pp_map = {}
        next_layer, next_expert = 0, 0
        glob_experts_tp_size = design.get("glob_experts_tp_size", None)

        for d in design.get("nodes", [{}]):
            (
                node_id,
                n_layers,
                n_experts,
                pp_size,
                ep_size,
                experts_tp_size,
                attn_tp_size,
            ) = [
                d.get(k)
                for k in [
                    "node_id",
                    "n_layers",
                    "n_experts",
                    "pp_size",
                    "ep_size",
                    "experts_tp_size",
                    "attn_tp_size",
                ]
            ]

            devices = get_devices(
                glob_experts_tp_size or experts_tp_size or ep_size or pp_size,
                None if glob_experts_tp_size else node_id,
            )
            attn_devices = get_devices(attn_tp_size, node_id)
            pp_bin_size = ceildiv(
                n_layers or model_config["num_hidden_layers"], pp_size or 1
            )
            ep_bin_size = ceildiv(
                n_experts or model_config["num_local_experts"], ep_size or 1
            )
            ffn_step = ceildiv(
                model_config["intermediate_size"],
                glob_experts_tp_size or experts_tp_size or 1,
            )
            for li in range(
                next_layer,
                (
                    next_layer + n_layers
                    if n_layers
                    else model_config["num_hidden_layers"]
                ),
            ):
                dis = devices
                if pp_size is not None:
                    dis = [devices[(li - next_layer) // pp_bin_size]]
                for ei in range(
                    next_expert,
                    (
                        next_expert + n_experts
                        if n_experts
                        else model_config["num_local_experts"]
                    ),
                ):
                    if ep_size is not None:
                        dis = [devices[(ei - next_expert) // ep_bin_size]]
                    expert_map[f"{li}-{ei}"] = (ffn_step, dis)

                if attn_devices:
                    attn_tp_map.setdefault(str(li), []).append(attn_devices)
                if pp_size:
                    non_expert_pp_map[str(li)] = dis
                elif n_layers:
                    non_expert_pp_map[str(li)] = devices

            next_layer += n_layers or 0
            next_expert += n_experts or 0

        assert next_layer == 0 or next_layer == model_config["num_hidden_layers"]
        assert next_expert == 0 or next_expert == model_config["num_local_experts"]
        return model_config, expert_map, attn_tp_map, non_expert_pp_map

    def load_weights(self) -> dict:
        weight_files = glob.glob(str(self.model_path / "consolidated.*.pt"))
        weights = {}
        for wf in weight_files:
            weights.update(torch.load(wf, weights_only=True, mmap=True))
        return weights

    def partition_expert_weights(self, ws: dict) -> None:
        interm_dim = self.model_config["intermediate_size"]
        partitions = {}

        for li in range(self.model_config["num_hidden_layers"]):
            w1: torch.Tensor = ws.pop(f"layers.{li}.block_sparse_moe.w1")
            w2: torch.Tensor = ws.pop(f"layers.{li}.block_sparse_moe.w2")
            w3: torch.Tensor = ws.pop(f"layers.{li}.block_sparse_moe.w3")

            for wi, w in enumerate([w1, w2, w3]):

                for ei, expert_slice in enumerate(torch.split(w, interm_dim)):
                    step, devices = self.expert_map[f"{li}-{ei}"]

                    for di, tp_slice in zip(devices, torch.split(expert_slice, step)):
                        sk = f"{li}.{ei}.w{wi + 1}"
                        partitions.setdefault(di, {})[sk] = tp_slice.clone()

        for di, partition in partitions.items():
            torch.save(partition, self.output_path / f"experts-{di}.pt")

        return ws

    def partition_non_expert_weights(self, ws: dict) -> None:
        n_model_layers = self.model_config["num_hidden_layers"]
        for li in range(n_model_layers):
            ws[f"layers.{li}.feed_forward.gate.weight"] = ws.pop(
                f"layers.{li}.block_sparse_moe.gate.weight"
            )

        if not self.attn_tp_map and not self.non_expert_pp_map:
            torch.save(ws, self.output_path / f"non-experts.pt")
            return

        n_attn_heads = self.model_config["num_attention_heads"]
        n_kv_heads = self.model_config["num_key_value_heads"]
        model_dim = self.model_config["hidden_size"]
        head_dim = model_dim // n_attn_heads
        attn_linear_wis = ["wq", "wk", "wv", "wo"]
        partitions = {}

        for li in range(n_model_layers):
            wq, wk, wv, wo, attn_norm, ffn_norm, gate = [
                ws.pop(wi)
                for wi in [
                    f"layers.{li}.attention.wq.weight",
                    f"layers.{li}.attention.wk.weight",
                    f"layers.{li}.attention.wv.weight",
                    f"layers.{li}.attention.wo.weight",
                    f"layers.{li}.attention_norm.weight",
                    f"layers.{li}.ffn_norm.weight",
                    f"layers.{li}.feed_forward.gate.weight",
                ]
            ]

            if self.attn_tp_map:
                for devices in self.attn_tp_map[str(li)]:
                    tp_size = len(devices)
                    assert n_attn_heads % tp_size == 0
                    assert n_kv_heads % tp_size == 0
                    qo_step = n_attn_heads * head_dim // tp_size
                    kv_step = n_kv_heads * head_dim // tp_size
                    steps = [qo_step, kv_step, kv_step, qo_step]

                    for wi, step, w in zip(attn_linear_wis, steps, [wq, wk, wv, wo.T]):

                        for di, w_slice in zip(devices, torch.split(w, step)):
                            partitions.setdefault(di, {})[
                                f"layers.{li}.attention.{wi}.weight"
                            ] = (w_slice.clone() if wi != "wo" else w_slice.T.clone())
            else:
                for di in self.non_expert_pp_map[str(li)]:
                    for wi, w in zip(attn_linear_wis, [wq, wk, wv, wo]):
                        partitions.setdefault(di, {})[
                            f"layers.{li}.attention.{wi}.weight"
                        ] = w

            non_attn_dest = self.non_expert_pp_map.get(str(li), [])
            if not non_attn_dest:
                for devices in self.attn_tp_map[str(li)]:
                    non_attn_dest.extend(devices)

            for di in non_attn_dest:
                partitions[di][f"layers.{li}.attention_norm.weight"] = attn_norm
                partitions[di][f"layers.{li}.ffn_norm.weight"] = ffn_norm
                partitions[di][f"layers.{li}.feed_forward.gate.weight"] = gate

            if li == 0:
                w_embed = ws.pop("tok_embeddings.weight")
                for di in non_attn_dest:
                    partitions[di]["tok_embeddings.weight"] = w_embed
            elif li == n_model_layers - 1:
                w_norm = ws.pop("norm.weight")
                w_output = ws.pop("output.weight")
                for di in non_attn_dest:
                    partitions[di]["norm.weight"] = w_norm
                    partitions[di]["output.weight"] = w_output

        for di, partition in partitions.items():
            torch.save(partition, self.output_path / f"non-experts-{di}.pt")

    def start(self) -> None:
        ws = self.partition_expert_weights(self.load_weights())
        logging.info("finished partitioning expert weights")
        self.partition_non_expert_weights(ws)
        logging.info("finished partitioning non-expert weights")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--design-path", type=str)
    parser.add_argument("--output-path", type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    weights_partitioner = Partitioner(
        args.model_path, args.design_path, args.output_path
    )
    weights_partitioner.start()
