"""
Partitioner designer for mixtral-8x7b weights from
https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
"""

from pathlib import Path
import argparse
import json
import logging
import pprint


class Partitioner:

    def __init__(self, model_path: str, design_path: str) -> None:
        self.model_path = Path(model_path)
        self.design_path = Path(design_path)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--design-path", type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    weights_partitioner = Partitioner(args.model_path, args.design_path)
    pprint.pp(weights_partitioner.expert_map)
    pprint.pp(weights_partitioner.attn_map)
