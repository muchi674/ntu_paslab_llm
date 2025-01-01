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


class Partitioner:

    def __init__(self, model_path: str, design_path: str, output_path: str) -> None:
        self.model_path = Path(model_path)
        self.design_path = Path(design_path)
        self.output_path = Path(output_path) if output_path else self.model_path
        self.model_config, self.expert_map, self.attn_map = self.get_configs()

    def get_configs(self) -> tuple[dict, dict]:
        with open(self.model_path / "config.json", "r") as model_config_file:
            model_config = json.load(model_config_file)
        with open(self.design_path, "r") as design_file:
            design = json.load(design_file)

        expert_map = {}
        attn_map = {}
        next_layer, next_expert = 0, 0

        for d in design["design"]:
            node_id, n_layers, n_experts, tp_size = (
                d.get("node_id"),
                d.get("n_layers"),
                d.get("n_experts"),
                d["tp_size"],
            )

            if len(design["design"]) > 1:
                assert node_id is not None
            # TODO: consider combining expert + pipeline parallelism
            assert (n_layers is None) or (n_experts is None)

            ffn_step = -(model_config["intermediate_size"] // -tp_size)  # ceil division
            partitions = []
            for ti in range(tp_size):
                if node_id is None:
                    name = str(ti)
                else:
                    name = f"{node_id}-{ti}"
                partitions.append(name)
            m = (ffn_step, partitions)

            for li in range(
                next_layer,
                (
                    next_layer + n_layers
                    if n_layers
                    else model_config["num_hidden_layers"]
                ),
            ):
                for ei in range(
                    next_expert,
                    (
                        next_expert + n_experts
                        if n_experts
                        else model_config["num_local_experts"]
                    ),
                ):
                    expert_map[f"{li}-{ei}"] = m

                if design.get("split_attn_weights", False):
                    attn_map.setdefault(str(li), []).append(partitions)

            next_layer += n_layers or 0
            next_expert += n_experts or 0

        assert next_layer == 0 or next_layer == model_config["num_hidden_layers"]
        assert next_expert == 0 or next_expert == model_config["num_local_experts"]
        return model_config, expert_map, attn_map

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
                    step, pks = self.expert_map[f"{li}-{ei}"]

                    for pk, tp_slice in zip(pks, torch.split(expert_slice, step)):
                        sk = f"{li}.{ei}.w{wi + 1}"
                        partitions.setdefault(pk, {})[sk] = tp_slice.clone()

        for pk, partition in partitions.items():
            torch.save(partition, self.output_path / f"experts-{pk}.pt")

        logging.info("finished partitioning expert weights")
        return ws

    def partition_non_expert_weights(self, ws: dict) -> None:
        for li in range(self.model_config["num_hidden_layers"]):
            ws[f"layers.{li}.feed_forward.gate.weight"] = ws.pop(
                f"layers.{li}.block_sparse_moe.gate.weight"
            )

        if self.attn_map:
            non_parallel_ks = ["tok_embeddings.weight", "norm.weight", "output.weight"]
            for li in range(self.model_config["num_hidden_layers"]):
                non_parallel_ks.append(f"layers.{li}.attention_norm.weight")
                non_parallel_ks.append(f"layers.{li}.ffn_norm.weight")
                non_parallel_ks.append(f"layers.{li}.feed_forward.gate.weight")
            non_parallel_ws = {k: ws.pop(k) for k in non_parallel_ks}

            n_attn_heads = self.model_config["num_attention_heads"]
            n_kv_heads = self.model_config["num_key_value_heads"]
            model_dim = self.model_config["hidden_size"]
            head_dim = model_dim // n_attn_heads
            partitions = {}

            for li in range(self.model_config["num_hidden_layers"]):
                wq: torch.Tensor = ws.pop(f"layers.{li}.attention.wq.weight")
                wk: torch.Tensor = ws.pop(f"layers.{li}.attention.wk.weight")
                wv: torch.Tensor = ws.pop(f"layers.{li}.attention.wv.weight")
                wo: torch.Tensor = ws.pop(f"layers.{li}.attention.wo.weight").T

                for pks in self.attn_map[str(li)]:
                    tp_size = len(pks)
                    assert n_attn_heads % tp_size == 0
                    assert n_kv_heads % tp_size == 0
                    q_step = n_attn_heads * head_dim // tp_size
                    kv_step = n_kv_heads * head_dim // tp_size
                    o_step = model_dim // tp_size
                    steps = [q_step, kv_step, kv_step, o_step]

                    for wi, step, w in zip(
                        ["wq", "wk", "wv", "wo"], steps, [wq, wk, wv, wo]
                    ):

                        for pk, w_slice in zip(pks, torch.split(w, step)):
                            partitions.setdefault(pk, {})[
                                f"layers.{li}.attention.{wi}.weight"
                            ] = (w_slice.clone() if wi != "wo" else w_slice.T.clone())

            for pk, partition in partitions.items():
                partition.update(non_parallel_ws)
                torch.save(partition, self.output_path / f"non-experts-{pk}.pt")

        else:
            torch.save(ws, self.output_path / f"non-experts.pt")

        logging.info("finished partitioning non-expert weights")
        return ws

    def start(self) -> None:
        ws = self.partition_expert_weights(self.load_weights())
        self.partition_non_expert_weights(ws)


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
