from pathlib import Path
import argparse
import gc
import json
import logging

import torch


class WeightsPreprocessor:

    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self.config = None

    def get_model_configs(self):
        try:
            with open(self.model_path / "params.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Config file not found in {self.model_path}")
            raise

    def load_weights(self) -> dict[str, torch.Tensor]:
        state_dict = {
            "tok_embeddings.weight": [],
            "norm.weight": [],
            "output.weight": [],
        }
        for li in range(self.config["n_layers"]):
            for k in ["q", "k", "v", "o"]:
                state_dict[f"layers.{li}.attention.w{k}.weight"] = []
            for k in range(1, 4):
                state_dict[f"layers.{li}.feed_forward.w{k}.weight"] = []
            for k in ["attention", "ffn"]:
                state_dict[f"layers.{li}.{k}_norm.weight"] = []

        for ckpt_path in sorted(self.model_path.glob("*.pth")):
            ws: dict[str, torch.Tensor] = torch.load(
                ckpt_path,
                map_location=torch.device("cpu"),
                weights_only=True,
                mmap=True,
            )
            for k, v in ws.items():
                state_dict[k].append(v)

        return state_dict

    def process_weights(self, state_dict: dict[str, torch.Tensor]) -> None:

        def cat_shards(k: str, dim: int = 0):
            state_dict[k] = torch.cat(state_dict[k], dim=dim)

        def disc_replicas(k: str):
            state_dict[k] = state_dict[k][0]

        for k in ["tok_embeddings.weight", "output.weight"]:
            cat_shards(k)
        disc_replicas("norm.weight")

        gc.collect()

        for li in range(self.config["n_layers"]):
            for k in ["q", "k", "v", "o"]:
                cat_shards(
                    f"layers.{li}.attention.w{k}.weight", dim=1 if k == "o" else 0
                )
            for k in range(1, 4):
                cat_shards(
                    f"layers.{li}.feed_forward.w{k}.weight", dim=1 if k == 2 else 0
                )
            for k in ["attention", "ffn"]:
                disc_replicas(f"layers.{li}.{k}_norm.weight")

            gc.collect()

        filename = "world_size_1_procr_0.pt"
        torch.save(state_dict, self.model_path / filename)
        logging.info(f"saved processed weights to {filename}")

    def start(self) -> None:
        self.config = self.get_model_configs()
        state_dict = self.load_weights()
        self.process_weights(state_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    weights_preprocessor = WeightsPreprocessor(args.model_path)
    weights_preprocessor.start()
