import argparse
import json
import logging
import time
from pathlib import Path

import safetensors.torch
import torch
from numpy.random import default_rng
from torch import nn


def get_json(file_path: Path) -> dict:
    try:
        with open(file_path, "r") as f:
            res = json.load(f)
    except FileNotFoundError:
        logging.error(f"{file_path} not found")
        raise

    return res


def main(model_path: str):

    model_path = Path(model_path)
    configs = get_json(model_path / "config.json")
    tensor_index = get_json(model_path / "model.safetensors.index.json")
    weight_filenames = set()
    weights = {}

    for k, v in tensor_index["weight_map"].items():
        if "block_sparse_moe.experts" in k:
            weight_filenames.add(v)

    for filename in weight_filenames:
        for k, v in safetensors.torch.load_file(
            model_path / filename, device="cpu"
        ).items():
            if "block_sparse_moe.experts" in k:
                weights[k] = v

    rng = default_rng(seed=0)
    selection = []
    for _ in range(configs["num_hidden_layers"]):
        selection.append(rng.choice(8, size=2, replace=False).tolist())

    gpu_0 = torch.device("cuda:0")
    x = torch.ones(1, configs["hidden_size"], dtype=torch.bfloat16, device=gpu_0)
    ys = []
    N = 64

    # compute everything on CPU
    # tic = time.perf_counter()
    # for _ in range(N):
    #     for li in range(configs["num_hidden_layers"]):
    #         x = x.to("cpu")
    #         for e in selection[li]:
    #             w1 = weights[f"model.layers.{li}.block_sparse_moe.experts.{e}.w1.weight"]
    #             w2 = weights[f"model.layers.{li}.block_sparse_moe.experts.{e}.w2.weight"]
    #             w3 = weights[f"model.layers.{li}.block_sparse_moe.experts.{e}.w3.weight"]
    #             ys.append((nn.functional.silu(x @ w1.T) * (x @ w3.T)) @ w2.T)
    #         x = x.to(gpu_0)
    # lat = time.perf_counter() - tic
    # logging.info(f"finished MoE computations in {lat} secs")
    # logging.info(f"each layer took {lat / N / configs['num_hidden_layers']} secs")

    # compute half of the layers on GPU
    # for li in range(configs["num_hidden_layers"]):
    #     for e in selection[li]:
    #         for wi in ["w1", "w2", "w3"]:
    #             weights[
    #                 f"model.layers.{li}.block_sparse_moe.experts.{e}.{wi}.weight"
    #             ] = weights[
    #                 f"model.layers.{li}.block_sparse_moe.experts.{e}.{wi}.weight"
    #             ].to(
    #                 gpu_0
    #             )

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)

    # start.record()

    # for _ in range(N):
    #     for li in range(configs["num_hidden_layers"]):
    #         for e in selection[li]:
    #             w1 = weights[
    #                 f"model.layers.{li}.block_sparse_moe.experts.{e}.w1.weight"
    #             ]
    #             w2 = weights[
    #                 f"model.layers.{li}.block_sparse_moe.experts.{e}.w2.weight"
    #             ]
    #             w3 = weights[
    #                 f"model.layers.{li}.block_sparse_moe.experts.{e}.w3.weight"
    #             ]
    #             x = ((nn.functional.silu(x @ w1.T) * (x @ w3.T)) @ w2.T) / 1000

    # end.record()
    # torch.cuda.synchronize()
    # lat = start.elapsed_time(end)
    # print(f"finished MoE computations in {lat} ms")
    # print(f"each layer took {start.elapsed_time(end) / N / configs['num_hidden_layers']} ms")
    # finished MoE computations in 1606.722412109375 ms
    # each layer took 0.7845324277877808 ms

    # move weights to GPU
    tic = time.perf_counter()

    for li in range(configs["num_hidden_layers"]):
        for e in selection[li]:
            for wi in ["w1", "w2", "w3"]:
                weights[
                    f"model.layers.{li}.block_sparse_moe.experts.{e}.{wi}.weight"
                ] = weights[
                    f"model.layers.{li}.block_sparse_moe.experts.{e}.{wi}.weight"
                ].to(
                    gpu_0
                )

    print(
        f"to gpu avg lat per layer: {(time.perf_counter() - tic) / configs['num_hidden_layers'] / 2} seconds"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    main(args.model_path)
