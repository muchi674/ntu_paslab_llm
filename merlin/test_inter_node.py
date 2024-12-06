import os

import torch
import torch.distributed as dist

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])


def run():
    device = torch.device(f"cuda:{LOCAL_RANK}")
    tensor = torch.ones((1, 4096), dtype=torch.bfloat16, device=device)
    group = dist.new_group(list(range(WORLD_SIZE)))
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print(tensor)


def init_processes():
    dist.init_process_group("nccl", rank=WORLD_RANK, world_size=WORLD_SIZE)
    run()
    dist.destroy_process_group()


init_processes()
# torchrun --nnodes=2 --node-rank=0 --nproc-per-node=2 --master-addr=10.10.10.1 --master-port=9091 test_inter_node.py
