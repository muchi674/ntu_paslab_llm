import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])

def expert(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor, group):
    """Simple collective communication."""
    # res = (F.silu(x @ w1.T) * (x @ w3.T)) @ w2
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)

def run():
    device = torch.device(f"cuda:{LOCAL_RANK}")
    x = torch.ones((1, 4096), dtype=torch.bfloat16, device=device)
    w1 = torch.ones((14336, 4096), dtype=torch.bfloat16, device=device) * (WORLD_RANK + 1)
    w2 = torch.ones((14336, 4096), dtype=torch.bfloat16, device=device) * (WORLD_RANK + 2)
    w3 = torch.ones((14336, 4096), dtype=torch.bfloat16, device=device) * (WORLD_RANK + 3)
    group = dist.new_group(list(range(WORLD_SIZE)))

    # warmup
    for _ in range(1000):
        expert(x, w1, w2, w3, group)

    tic = time.time()

    for _ in range(100000):
        expert(x, w1, w2, w3, group)

    print(f"AVG run latency: {((time.time() - tic) * 1000) / 100000} ms")

def init_processes():
    dist.init_process_group("nccl", rank=WORLD_RANK, world_size=WORLD_SIZE)
    run()
    dist.destroy_process_group()

# torchrun --nnodes=2 --node-rank=0 --nproc-per-node=2 --master-addr=10.10.10.1 --master-port=9091 test_inter_node.py
init_processes()
