import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_process(
    rank: int,
    world_size: int,
):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "9091"
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, device_id=device
    )

    n_tokens = 1
    hidden_dim = 4096
    intermediate_dim = 14336 // 4
    x = torch.zeros((n_tokens, hidden_dim), dtype=torch.bfloat16, device=device)
    w1 = torch.rand((hidden_dim, intermediate_dim), dtype=torch.bfloat16, device=device)
    w2 = torch.rand((intermediate_dim, hidden_dim), dtype=torch.bfloat16, device=device)

    n_tests = 32
    n_warmups = 32
    n_layers = 32
    n_forward = 8

    for i in range(n_warmups):
        for j in range(n_layers):
            for k in range(n_forward): 
                x = x @ w1
                x = x @ w2
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
    
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.synchronize(device=device)
    for i in range(n_tests):
        for j in range(n_layers):
            for k in range(n_forward):
                x = x @ w1
                x = x @ w2
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(device=device)
        dist.barrier()
    torch.cuda.cudart().cudaProfilerStop()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    processes = []
    mp.set_start_method("spawn")

    for rank in range(world_size):
        p = mp.Process(
            target=init_process,
            args=(
                rank,
                world_size,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
