import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])


def run(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor, group):
    res = (F.silu(x @ w1.T) * (x @ w3.T)) @ w2.T
    dist.all_reduce(res, op=dist.ReduceOp.SUM, group=group)


def init_processes():
    dist.init_process_group(
        "nccl",
        rank=WORLD_RANK,
        world_size=WORLD_SIZE,
    )

    device = torch.device(f"cuda:{LOCAL_RANK}")
    group = dist.new_group(list(range(WORLD_SIZE)))
    x = torch.ones((1, 4096), dtype=torch.bfloat16, device=device)
    w1 = (
        torch.ones((14336, 4096), dtype=torch.bfloat16, device=device)
        * (WORLD_RANK + 1)
        / 10**6
    )
    w2 = (
        torch.ones((4096, 14336), dtype=torch.bfloat16, device=device)
        * (WORLD_RANK + 2)
        / 10**6
    )
    w3 = (
        torch.ones((14336, 4096), dtype=torch.bfloat16, device=device)
        * (WORLD_RANK + 3)
        / 10**6
    )

    # warmup
    for _ in range(100):
        run(x, w1, w2, w3, group)

    tic = time.time()

    for _ in range(10000):
        run(x, w1, w2, w3, group)

    print(f"AVG run latency: {((time.time() - tic) * 1000) / 10000} ms")
    dist.destroy_process_group()


if __name__ == "__main__":
    init_processes()
    # torchrun --nnodes=2 --node-rank=0 --nproc-per-node=2 --master-addr=10.10.10.1 --master-port=9091 test_inter_node.py
