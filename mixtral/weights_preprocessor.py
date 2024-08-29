from pathlib import Path
import argparse
import gc
import json
import logging

import safetensors.torch

import torch


class WeightsPreprocessor:

    def __init__(self, input_path: str, output_path: str, hf: bool) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.hf = hf
        self.config = None

    def get_hf_model_configs(self):
        try:
            with open(self.input_path / "config.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Config file not found in {self.input_path}")
            raise

    def load_hf_weights(self) -> dict:
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

    def process_hf_experts(self, ws: dict) -> None:
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

    def process_hf_non_experts(self, ws: dict) -> None:
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

        safetensors.torch.save_file(
            non_experts, self.output_path / f"non-experts.safetensors"
        )
        logging.info("finished processing non-expert weights")

        return ws

    def get_pth_model_configs(self):
        try:
            with open(self.input_path / "params.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Config file not found in {self.input_path}")
            raise

    def load_pth_weights(self) -> dict:
        return torch.load(self.input_path / "consolidated.00.pth", mmap=True)

    def process_pth_experts(self, ws: dict) -> None:
        experts = {}
        for li in range(self.config["n_layers"]):
            for ei in range(self.config["moe"]["num_experts"]):
                w1 = ws.pop(f"layers.{li}.feed_forward.experts.{ei}.w1.weight")
                w2 = ws.pop(f"layers.{li}.feed_forward.experts.{ei}.w2.weight")
                w3 = ws.pop(f"layers.{li}.feed_forward.experts.{ei}.w3.weight")
                experts[f"{li}.{ei}"] = torch.stack((w1, w2.T, w3), dim=0)

        torch.save(experts, self.output_path / "experts.pt")
        logging.info("finished processing expert weights")

        return ws

    def process_pth_non_experts(self, ws: dict) -> None:
        torch.save(ws, self.output_path / f"non-experts.pt")
        logging.info("finished processing non-expert weights")
        return ws

    def start(self) -> None:
        if self.hf:
            self.config = self.get_hf_model_configs()
            ws = self.process_hf_experts(self.load_hf_weights())
            gc.collect()
            self.process_hf_non_experts(ws)
        else:
            self.config = self.get_pth_model_configs()
            ws = self.process_pth_experts(self.load_pth_weights())
            self.process_pth_non_experts(ws)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--hf", action="store_true")  # uses pth weights by default
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    weights_preprocessor = WeightsPreprocessor(
        args.input_path, args.output_path, args.hf
    )
    weights_preprocessor.start()
