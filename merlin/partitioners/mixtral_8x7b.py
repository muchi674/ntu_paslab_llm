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

    def __init__(self, model_path: str, design_path: str) -> None:
        self.model_path = Path(model_path)
        self.design_path = Path(design_path)
        self.model_config, self.design = self.get_configs()

    def get_configs(self) -> tuple[dict, dict]:
        with open(self.model_path / "config.json", "r") as model_config_file:
            model_config = json.load(model_config_file)
        with open(self.design_path, "r") as design_file:
            design = json.load(design_file)
        return model_config, design

    def load_weights(self) -> dict:
        weight_files = glob.glob(str(self.model_path / "consolidated.*.pt"))
        weights = {}
        for wf in weight_files:
            weights.update(torch.load(wf, weights_only=True, mmap=True))
        return weights

    def partition_expert_weights(self, ws: dict) -> None:
        """current implementation only supports tensor parallelism"""
        tp_size = self.design["tp_size"]
        interm_dim = self.model_config["intermediate_size"]

        step = -(interm_dim // -tp_size)  # ceil division
        tp_ranges = []
        for start in range(0, interm_dim, step):
            tp_ranges.append((start, min(start + step, interm_dim)))

        partitions = [{} for _ in range(tp_size)]

        for li in range(self.model_config["num_hidden_layers"]):
            w1: torch.Tensor = ws.pop(f"layers.{li}.block_sparse_moe.w1")
            w2: torch.Tensor = ws.pop(f"layers.{li}.block_sparse_moe.w2")
            w3: torch.Tensor = ws.pop(f"layers.{li}.block_sparse_moe.w3")

            for tp_range, partition in zip(tp_ranges, partitions):
                tpl, tpr = tp_range

                for wi, w in enumerate([w1, w2, w3]):
                    slices = []

                    for el in range(0, w1.shape[0], interm_dim):
                        slices.append(w[el : el + interm_dim][tpl:tpr])

                    partition[f"{li}.w{wi + 1}"] = torch.cat(slices, dim=0)

        for pi, partition in enumerate(partitions):
            torch.save(partition, self.model_path / f"experts-tp-{pi}.pt")

        logging.info("finished partitioning expert weights")
        return ws

    def bundle_non_expert_weights(self, ws: dict) -> None:
        for li in range(self.model_config["num_hidden_layers"]):
            ws[f"layers.{li}.feed_forward.gate.weight"] = ws.pop(
                f"layers.{li}.block_sparse_moe.gate.weight"
            )
        torch.save(ws, self.model_path / f"non-experts.pt")
        logging.info("finished bundling non-expert weights")
        return ws

    def start(self) -> None:
        ws = self.partition_expert_weights(self.load_weights())
        self.bundle_non_expert_weights(ws)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--design-path", type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    weights_partitioner = Partitioner(args.model_path, args.design_path)
    weights_partitioner.start()
