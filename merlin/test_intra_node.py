#!/home/muchichen/miniconda3/envs/merlin/bin/python
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(x: torch.Tensor, w: torch.Tensor, group):
    """Simple collective communication."""
    res = x @ w.T
    dist.all_reduce(res, op=dist.ReduceOp.SUM, group=group)


def init_process(rank, size, fn):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "9091"
    dist.init_process_group(backend="nccl", rank=rank, world_size=size)

    device = torch.device(f"cuda:{rank}")
    group = dist.new_group([0, 1])
    x = torch.ones((128, 4096), dtype=torch.bfloat16, device=device)
    w = torch.ones((14336, 4096), dtype=torch.bfloat16, device=device) * (rank + 1)
    for _ in range(10000):
        fn(x, w, group)


if __name__ == "__main__":
    world_size = 2
    processes = []
    mp.set_start_method("spawn")

    tic = time.time()

    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"AVG run latency: {((time.time() - tic) * 1000) / 10000} ms")
