import argparse
import json
import logging
import time
import concurrent.futures
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

    # def mlp(x: torch.Tensor, li: int, e: int, gpu: torch.device, compute_on_gpu: bool):
    #     w1 = weights[f"model.layers.{li}.block_sparse_moe.experts.{e}.w1.weight"]
    #     w2 = weights[f"model.layers.{li}.block_sparse_moe.experts.{e}.w2.weight"]
    #     w3 = weights[f"model.layers.{li}.block_sparse_moe.experts.{e}.w3.weight"]

    #     if compute_on_gpu:
    #         w1.to(gpu)
    #         w2.to(gpu)
    #         w3.to(gpu)
    #         x.to(gpu, copy=True)

    #     y = (nn.functional.silu(x @ w1.T) * (x @ w3.T)) @ w2.T
        
    #     if not compute_on_gpu:
    #         y.to(gpu)

    #     return y

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

    tic = time.perf_counter()

    # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as xcutor:
    #     for li in range(configs["num_hidden_layers"]):
    #         e0, e1 = selection[li]
    #         cpu_job = xcutor.submit(mlp, x, li, e0, gpu_0, True)
    #         gpu_job = xcutor.submit(mlp, x, li, e1, gpu_0, True)
    #         concurrent.futures.wait([gpu_job, cpu_job])
    #         ys.append(cpu_job.result() + gpu_job.result())

    # for li in range(configs["num_hidden_layers"] // 2):
    #     for e in selection[li]:
    #         w1 = weights[f"model.layers.{li}.block_sparse_moe.experts.{e}.w1.weight"]
    #         w2 = weights[f"model.layers.{li}.block_sparse_moe.experts.{e}.w2.weight"]
    #         w3 = weights[f"model.layers.{li}.block_sparse_moe.experts.{e}.w3.weight"]
    #         w1.to(gpu_0)
    #         w2.to(gpu_0)
    #         w3.to(gpu_0)

    for li in range(configs["num_hidden_layers"]):
        x.to("cpu")
        x.to(gpu_0)

    logging.info(f"finished MoE computations in {time.perf_counter() - tic} secs")
    print(len(ys))
    # print(ys[0].shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    main(args.model_path)
