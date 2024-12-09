#!/home/muchichen/miniconda3/envs/merlin/bin/python
import os
import time

from torch import nn
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor, group):
    """Simple collective communication."""
    # res = (nn.functional.silu(x @ w1.T) * (x @ w3.T)) @ w2
    res = x
    dist.all_reduce(res, op=dist.ReduceOp.SUM, group=group)


def init_process(rank, size, fn):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "9091"
    dist.init_process_group(backend="nccl", rank=rank, world_size=size)

    device = torch.device(f"cuda:{rank}")
    group = dist.new_group([0, 1])
    x = torch.ones((1, 32), dtype=torch.bfloat16, device=device)
    w1 = torch.ones((14336, 4096), dtype=torch.bfloat16, device=device) * (rank + 1)
    w2 = torch.ones((14336, 4096), dtype=torch.bfloat16, device=device) * (rank + 2)
    w3 = torch.ones((14336, 4096), dtype=torch.bfloat16, device=device) * (rank + 3)

    # warmup
    for _ in range(1000):
        fn(x, w1, w2, w3, group)

    tic = time.time()

    for _ in range(100000):
        fn(x, w1, w2, w3, group)

    print(f"AVG run latency: {((time.time() - tic) * 1000) / 100000} ms")
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    processes = []
    mp.set_start_method("spawn")

    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
