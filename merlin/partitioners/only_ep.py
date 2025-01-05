"""
Partitioner designer for mixtral-8x7b weights (original code, minimal modification)
Only does expert parallel, no tensor parallel, no non-expert splitting.
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

       
        self.model_config, self.expert_map, self.attn_tp_map, self.non_expert_pp_map = (
            self.get_configs()
        )

    def get_configs(self) -> tuple[dict, dict, dict, dict]:
        with open(self.model_path / "config.json", "r") as model_config_file:
            model_config = json.load(model_config_file)
        with open(self.design_path, "r") as design_file:
            design = json.load(design_file)

        expert_map = {}
        
        attn_tp_map = {}
        non_expert_pp_map = {}

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
            assert n_layers is None or n_layers > 0
            assert n_experts is None or n_experts > 0
            
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

    def partition_expert_weights(self, ws: dict) -> dict:
        

        interm_dim = self.model_config["intermediate_size"]

       
        parted_experts = {}

        for li in range(self.model_config["num_hidden_layers"]):
            w1: torch.Tensor = ws.pop(f"layers.{li}.block_sparse_moe.w1")
            w2: torch.Tensor = ws.pop(f"layers.{li}.block_sparse_moe.w2")
            w3: torch.Tensor = ws.pop(f"layers.{li}.block_sparse_moe.w3")

            
            for wi, w in enumerate([w1, w2, w3], start=1):
                
                slices = torch.split(w, interm_dim)

                for ei, expert_slice in enumerate(slices):
                    
                    _, pks = self.expert_map[f"{li}-{ei}"]
                    
                    parted_experts.setdefault(ei, {})[
                        f"layers.{li}.block_sparse_moe.w{wi}"
                    ] = expert_slice.clone()

        
        for g_ei, data_dict in parted_experts.items():
            fname = f"experts-{g_ei}.pt"
            torch.save(data_dict, self.output_path / fname)

        return ws  


    def partition_non_expert_weights(self, ws: dict) -> None:
       
        n_model_layers = self.model_config["num_hidden_layers"]
        
        for li in range(n_model_layers):
            ws[f"layers.{li}.feed_forward.gate.weight"] = ws.pop(
                f"layers.{li}.block_sparse_moe.gate.weight"
            )

        torch.save(ws, self.output_path / f"non-experts.pt")

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
