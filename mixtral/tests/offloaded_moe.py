import argparse
import gc
import json
import logging
import time
from pathlib import Path
from statistics import mean

import safetensors.torch
import torch
from torch import nn

# DEPRECATED: for raw weights
# from numpy.random import default_rng

# tensor_index = get_json(model_path / "model.safetensors.index.json")
# weight_filenames = set()
# weights = {}

# for k, v in tensor_index["weight_map"].items():
#     if "block_sparse_moe.experts" in k:
#         weight_filenames.add(v)

# for filename in weight_filenames:
#     for k, v in safetensors.torch.load_file(
#         model_path / filename, device="cpu"
#     ).items():
#         if "block_sparse_moe.experts" in k:
#             weights[k] = v

# gc.collect()

# rng = default_rng(seed=0)
# selection = []
# for _ in range(configs["num_hidden_layers"]):
#     selection.append(rng.choice(8, size=2, replace=False).tolist())

# # compute everything on CPU
# tic = time.perf_counter()
# for _ in range(N):
#     for li in range(configs["num_hidden_layers"]):
#         x = x.to("cpu")
#         for e in range(2):
#             w1 = weights[f"model.layers.{li}.block_sparse_moe.experts.{e}.w1.weight"]
#             w2 = weights[f"model.layers.{li}.block_sparse_moe.experts.{e}.w2.weight"]
#             w3 = weights[f"model.layers.{li}.block_sparse_moe.experts.{e}.w3.weight"]
#             x = ((nn.functional.silu(x @ w1.T) * (x @ w3.T)) @ w2.T) / 1000
#         x = x.to(gpu_0)
# lat = time.perf_counter() - tic
# logging.info(f"finished MoE computations in {lat} secs")
# logging.info(f"each layer took {lat / N / configs['num_hidden_layers']} secs")
# INFO:root:finished MoE computations in 9.442421261002892 secs
# INFO:root:each layer took 0.014753783220317018 secs


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
    # configs = get_json(model_path / "config.json")
    gpu_0 = torch.device("cuda:0")
    # ws = safetensors.torch.load_file(model_path / "experts.safetensors", device="cpu")
    ws: dict[str, torch.Tensor] = torch.load(
        model_path / "experts.pt", map_location=torch.device("cpu"), mmap=True
    )

    # for k in ws:
    #     ws[k] = ws[k].pin_memory()
    # gc.collect()

    # x = torch.ones(1, configs["hidden_size"], dtype=torch.bfloat16, device=gpu_0)
    # x1 = torch.ones(1, configs["hidden_size"], dtype=torch.bfloat16, device=gpu_0) * 2
    # x2 = torch.ones(1, configs["hidden_size"], dtype=torch.bfloat16, device=gpu_0) * 3
    ys = []
    N = 10

    # -------------------------------------------------------------------------------------

    # cpu computation with stacked weights
    # tic = time.perf_counter()
    # for _ in range(N):
    #     for li in range(configs["num_hidden_layers"]):
    #         x = x.to("cpu")
    #         for ei in range(2):
    #             w = ws[f"{li}.{ei}"]
    #             ys.append((nn.functional.silu(x @ w[0].T) * (x @ w[2].T)) @ w[1])
    #         x = x.to(gpu_0)
    # lat = time.perf_counter() - tic
    # logging.info(f"finished MoE computations in {lat} secs")
    # logging.info(f"each layer took {lat / N / configs['num_hidden_layers']} secs")

    # -------------------------------------------------------------------------------------

    # # move stacked weights to GPU
    # for li in range(configs["num_hidden_layers"]):
    #     for ei in range(2):
    #         ws[f"{li}.{ei}"] = ws[f"{li}.{ei}"].to(gpu_0, non_blocking=True)

    # # wait for weights copy to finish
    # torch.cuda.synchronize(device=gpu_0)

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)

    # # GPU computation on stacked weights
    # start.record()

    # for _ in range(N):
    #     for li in range(configs["num_hidden_layers"]):
    #         for ei in range(2):
    #             w = ws[f"{li}.{ei}"]
    #             ys.append((nn.functional.silu(x @ w[0].T) * (x @ w[2].T)) @ w[1])
    #         torch.cuda.synchronize()

    # end.record()
    # torch.cuda.synchronize()
    # lat = start.elapsed_time(end)
    # print(f"finished MoE computations in {lat} ms")
    # print(f"each layer took {start.elapsed_time(end) / N / configs['num_hidden_layers']} ms")

    # -------------------------------------------------------------------------------------

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)

    # # overlap weights transfer to GPU memory with CPU computation
    # start.record()

    # for _ in range(N):
    #     for li in range(configs["num_hidden_layers"]):
    #         x = x.to("cpu")
    #         ws[f"{li}.3"].to(gpu_0, non_blocking=True)
    #         for ei in range(3):
    #             w = ws[f"{li}.{ei}"]
    #             ys.append((nn.functional.silu(x @ w[0].T) * (x @ w[2].T)) @ w[1])
    #         torch.cuda.synchronize(device=gpu_0)
    #         x = x.to(gpu_0)

    # end.record()
    # torch.cuda.synchronize(device=gpu_0)
    # lat = start.elapsed_time(end)
    # print(f"finished MoE computations in {lat} ms")
    # print(
    #     f"each layer took {start.elapsed_time(end) / N / configs['num_hidden_layers']} ms"
    # )

    # -------------------------------------------------------------------------------------

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)

    # start.record()

    # for _ in range(N):
    #     for li in range(configs["num_hidden_layers"]):
    #         cpu_x = x.to("cpu")
    #         gpu_w = ws[f"{li}.6"].to(gpu_0, non_blocking=True)
    #         ys.append(
    #             (nn.functional.silu(x @ gpu_w[0].T) * (x @ gpu_w[2].T)) @ gpu_w[1]
    #         )
    #         ys.append(
    #             (nn.functional.silu(x1 @ gpu_w[0].T) * (x1 @ gpu_w[2].T)) @ gpu_w[1]
    #         )
    #         ys.append(
    #             (nn.functional.silu(x2 @ gpu_w[0].T) * (x2 @ gpu_w[2].T)) @ gpu_w[1]
    #         )
    #         tmp = []
    #         for ei in range(6):
    #             w = ws[f"{li}.{ei}"]
    #             tmp.append((nn.functional.silu(cpu_x @ w[0].T) * (cpu_x @ w[2].T)) @ w[1])
    #         ys.append(torch.sum(torch.stack(tmp), dim=0).to(gpu_0))
    #         torch.cuda.synchronize(device=gpu_0)

    # end.record()
    # torch.cuda.synchronize(device=gpu_0)
    # lat = start.elapsed_time(end)
    # print(f"finished MoE computations in {lat} ms")
    # print(
    #     f"each layer took {start.elapsed_time(end) / N / configs['num_hidden_layers']} ms"
    # )

    # -------------------------------------------------------------------------------------
    n_warmups = 10
    x = torch.ones(2, 4096, dtype=torch.bfloat16, device="cpu")

    # warmup
    for _ in range(n_warmups):
        w = ws["0.0"]
        (nn.functional.silu(x @ w[0].T) * (x @ w[2].T)) @ w[1]

    latencies = []
    for _ in range(N):
        w = ws["0.0"]
        tic = time.perf_counter()
        # for _ in range(128):
        #     (nn.functional.silu(x @ w[0].T) * (x @ w[2].T)) @ w[1]
        (nn.functional.silu(x @ w[0].T) * (x @ w[2].T)) @ w[1]
        latencies.append((time.perf_counter() - tic) * 1000)

    print(f"avg latency: {mean(latencies)} ms")
    print(latencies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    main(args.model_path)
