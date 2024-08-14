import argparse
import json
import logging
import time
from pathlib import Path

import safetensors.torch
import torch
from torch import nn

# class FeedForward(nn.Module):
#     def __init__(self, args: TransformerArgs):
#         super().__init__()

#         self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
#         self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
#         self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

def get_json(file_path: Path) -> dict:
    try:
        with open(file_path, "r") as f:
            res = json.load(f)
    except FileNotFoundError:
        logging.error(f"{file_path} not found")
        raise

    return res

def load_model(model_path: str):
    model_path = Path(model_path)
    configs = get_json(model_path / "config.json")
    tensor_index = get_json(model_path / "model.safetensors.index.json")
    weight_filenames = set()
    weights = {}

    for k, v in tensor_index["weight_map"].items():
        if "block_sparse_moe.experts" in k:
            weight_filenames.add(v)

    for filename in weight_filenames:
        for k, v in safetensors.torch.load_file(filename, device="cpu").items():
            if "block_sparse_moe.experts" in k:
                weights[k] = v

    logging.info(len(weights))
    time.sleep(10)

    # assert (
    #     pt_model_file.exists() or safetensors_model_file.exists()
    # ), f"Make sure either {pt_model_file} or {safetensors_model_file} exists"
    # assert not (
    #     pt_model_file.exists() and safetensors_model_file.exists()
    # ), f"Both {pt_model_file} and {safetensors_model_file} cannot exist"

    # loaded = safetensors.torch.load_file(str(safetensors_model_file))

    # model.load_state_dict(loaded, assign=True, strict=True)

    # return model.to(device=device, dtype=dtype)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()

    load_model(args.model_path)
