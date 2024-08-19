import argparse
import gc
import json
import logging
import time
from pathlib import Path
from statistics import mean

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

    gc.collect()

    rng = default_rng(seed=0)
    selection = []
    for _ in range(configs["num_hidden_layers"]):
        # selection.append(rng.choice(8, size=2, replace=False).tolist())
        selection.append(list(range(6)))

    gpu_0 = torch.device("cuda:0")
    x = torch.ones(1, configs["hidden_size"], dtype=torch.bfloat16, device=gpu_0)
    ys = []
    N = 20

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
    # INFO:root:finished MoE computations in 9.442421261002892 secs
    # INFO:root:each layer took 0.014753783220317018 secs

    # -------------------------------------------------------------------------------------

    # move weights to and compute on GPU
    # tic = time.perf_counter()

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

    # lat = (time.perf_counter() - tic) / configs["num_hidden_layers"] / 2
    # print(f"to gpu avg lat per layer per expert: {lat} seconds")

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

    # -------------------------------------------------------------------------------------

    # move stacked weights to GPU
    # stacked = []

    # for li in range(configs["num_hidden_layers"]):
    #     w1 = weights[f"model.layers.{li}.block_sparse_moe.experts.0.w1.weight"]
    #     w2 = weights[f"model.layers.{li}.block_sparse_moe.experts.0.w2.weight"]
    #     w3 = weights[f"model.layers.{li}.block_sparse_moe.experts.0.w3.weight"]
    #     stacked.append(torch.stack((w1, w2.T, w3)))

    # tic = time.perf_counter()

    # for li in range(configs["num_hidden_layers"]):
    #     stacked[li] = stacked[li].to(gpu_0)

    # lat = (time.perf_counter() - tic) / configs["num_hidden_layers"]
    # print(f"to gpu avg lat per layer per expert: {lat} seconds")

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)

    # start.record()

    # for _ in range(N):
    #     for li in range(configs["num_hidden_layers"]):
    #         ws = stacked[li]
    #         ys.append((nn.functional.silu(x @ ws[0].T) * (x @ ws[2].T)) @ ws[1])
    #         torch.cuda.synchronize()

    # end.record()
    # torch.cuda.synchronize()
    # lat = start.elapsed_time(end)
    # print(f"finished MoE computations in {lat} ms")
    # print(f"each layer took {start.elapsed_time(end) / N / configs['num_hidden_layers']} ms")

    # -------------------------------------------------------------------------------------

    # overlap weights transfer to GPU memory with CPU compute
    # assumes num selected experts per layer = 6
    stacked = []

    for li in range(configs["num_hidden_layers"]):
        w1 = weights[f"model.layers.{li}.block_sparse_moe.experts.5.w1.weight"]
        w2 = weights[f"model.layers.{li}.block_sparse_moe.experts.5.w2.weight"]
        w3 = weights[f"model.layers.{li}.block_sparse_moe.experts.5.w3.weight"]
        stacked.append(torch.stack((w1, w2.T, w3)).pin_memory())

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for _ in range(N):
        for li in range(configs["num_hidden_layers"]):
            cpu_x = x.to("cpu")
            ws = stacked[li].to(gpu_0, non_blocking=True)
            for e in range(5):
                w1 = weights[f"model.layers.{li}.block_sparse_moe.experts.{e}.w1.weight"]
                w2 = weights[f"model.layers.{li}.block_sparse_moe.experts.{e}.w2.weight"]
                w3 = weights[f"model.layers.{li}.block_sparse_moe.experts.{e}.w3.weight"]
                ys.append((nn.functional.silu(cpu_x @ w1.T) * (cpu_x @ w3.T)) @ w2.T)
            # torch.cuda.synchronize()
            ys.append((nn.functional.silu(x @ ws[0].T) * (x @ ws[2].T)) @ ws[1])
            torch.cuda.synchronize()

    end.record()
    torch.cuda.synchronize()
    lat = start.elapsed_time(end)
    print(f"finished MoE computations in {lat} ms")
    print(f"each layer took {start.elapsed_time(end) / N / configs['num_hidden_layers']} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    main(args.model_path)
