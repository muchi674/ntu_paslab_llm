import argparse
import logging
import time
from pathlib import Path
from statistics import mean

import torch
from torch import nn

def test0():
    cpu = torch.device("cpu")
    gpu_0 = torch.device("cuda:0")

    N = 1000
    n_warmups = 20
    x = torch.ones((8192,), dtype=torch.bfloat16, device=cpu)
    w1 = torch.ones((8192, 8192), dtype=torch.bfloat16, device=cpu) * 2
    w2 = torch.ones((8192, 8192), dtype=torch.bfloat16, device=cpu) * 3
    w3 = torch.ones((1024, 8192), dtype=torch.bfloat16, device=cpu) * 4
    w4 = torch.ones((1024, 8192), dtype=torch.bfloat16, device=cpu) * 4

    # for _ in range(n_warmups):
    #     (nn.functional.silu(x @ w1.T) * (x @ w3.T)) @ w2.T

    # w0 = torch.ones((28672, 8192), dtype=torch.bfloat16, device=cpu).pin_memory()
    # w0 = torch.ones((3, 28672, 8192), dtype=torch.bfloat16, device=cpu).pin_memory()

    latencies = []
    for _ in range(N):
        tic = time.perf_counter()
        x @ w1.T, x @ w2.T, x @ w3.T, x @ w4.T
        # (nn.functional.silu(x @ w1.T) * (x @ w3.T)) @ w2.T
        # w0.to(gpu_0)
        latencies.append((time.perf_counter() - tic) * 1000)

    print(f"avg latency: {mean(latencies)} ms")
    print(latencies)

def main(model_path: str):

    model_path = Path(model_path)
    cpu = torch.device("cpu")
    gpu_0 = torch.device("cuda:0")
    # ws: dict[str, torch.Tensor] = torch.load(
    #     model_path / "consolidated.00.pth",
    #     map_location=torch.device("cpu"),
    #     weights_only=True,
    #     mmap=True,
    # )

    N = 100
    n_warmups = 20
    x = torch.ones((8192,), dtype=torch.bfloat16, device=cpu)
    # w1, w2, w3 = ws["layers.0.feed_forward.w1.weight"], ws["layers.0.feed_forward.w2.weight"], ws["layers.0.feed_forward.w3.weight"]

    # warmup
    # for _ in range(n_warmups):
    #     (nn.functional.silu(x @ w1.T) * (x @ w2.T)) @ w3.T

    # latencies = []
    # for _ in range(N):
    #     tic = time.perf_counter()
    #     (nn.functional.silu(x @ w1.T) * (x @ w2.T)) @ w3.T
    #     latencies.append((time.perf_counter() - tic) * 1000)

    # print(f"avg latency: {mean(latencies)} ms")
    # print(latencies)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str)
    # args = parser.parse_args()
    # logging.basicConfig(level=logging.INFO)

    # main(args.model_path)
    test0()
