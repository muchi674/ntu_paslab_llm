from pathlib import Path
import argparse
import gc
import json
import logging

import safetensors.torch

import torch


class WeightsPreprocessor:

    def __init__(self, input_path: str, output_path: str) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.config = self.get_model_configs()

    def get_model_configs(self):
        try:
            with open(self.input_path / "config.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Config file not found in {self.input_path}")
            raise

    def load_weights(self) -> dict:
        try:
            with open(self.input_path / "model.safetensors.index.json", "r") as f:
                metadata = json.load(f)
        except FileNotFoundError:
            logging.error(
                f"model.safetensors.index.json not found in {self.input_path}"
            )
            raise

        ws = {}
        for filename in set(metadata["weight_map"].values()):
            ws.update(
                safetensors.torch.load_file(self.input_path / filename, device="cpu")
            )
        return ws

    def process_experts(self, ws: dict) -> None:
        experts = {}
        for li in range(self.config["num_hidden_layers"]):
            for ei in range(self.config["num_local_experts"]):
                w1 = ws.pop(
                    f"model.layers.{li}.block_sparse_moe.experts.{ei}.w1.weight"
                )
                w2 = ws.pop(
                    f"model.layers.{li}.block_sparse_moe.experts.{ei}.w2.weight"
                )
                w3 = ws.pop(
                    f"model.layers.{li}.block_sparse_moe.experts.{ei}.w3.weight"
                )
                experts[f"{li}.{ei}"] = torch.stack((w1, w2.T, w3), dim=0)
            gc.collect()

        safetensors.torch.save_file(experts, self.output_path / f"experts.safetensors")
        logging.info("finished processing expert weights")

        return ws

    def process_non_experts(self, ws: dict) -> None:
        non_experts = {
            "tok_embeddings.weight": ws.pop("model.embed_tokens.weight"),
            "norm.weight": ws.pop("model.norm.weight"),
            "lm_head.weight": ws.pop("lm_head.weight"),
        }
        for li in range(self.config["num_hidden_layers"]):
            prefix = f"model.layers.{li}"
            pfx = prefix[6:]
            non_experts[f"{pfx}.attention_norm.weight"] = ws.pop(
                f"{prefix}.input_layernorm.weight"
            )
            non_experts[f"{pfx}.ffn_norm.weight"] = ws.pop(
                f"{prefix}.post_attention_layernorm.weight"
            )
            non_experts[f"{pfx}.feed_forward.gate.weight"] = ws.pop(
                f"{prefix}.block_sparse_moe.gate.weight"
            )
            for pi in ["q", "k", "v", "o"]:
                non_experts[f"{pfx}.attention.w{pi}.weight"] = ws.pop(
                    f"{prefix}.self_attn.{pi}_proj.weight"
                )

        safetensors.torch.save_file(non_experts, self.output_path / f"non-experts.safetensors")
        logging.info("finished processing non-expert weights")

        return ws

    def start(self) -> None:
        ws = self.process_experts(self.load_weights())
        gc.collect()
        ws = self.process_non_experts(ws)
        assert len(ws) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--output-path", type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    weights_preprocessor = WeightsPreprocessor(args.input_path, args.output_path)
    weights_preprocessor.start()
