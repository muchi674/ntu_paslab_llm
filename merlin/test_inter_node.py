import os

import torch
import torch.distributed as dist

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])


def run():
    tensor = torch.zeros(1)
    device = torch.device(f"cuda:{LOCAL_RANK}")
    tensor = tensor.to(device)

    if WORLD_RANK == 0:
        for rank_recv in range(1, WORLD_SIZE):
            dist.send(tensor=tensor, dst=rank_recv)
            print(f"rank_0 sent data to rank_{rank_recv}")
    else:
        dist.recv(tensor=tensor, src=0)
        print(f"rank_{WORLD_RANK} has received data from rank_0")


def init_processes():
    dist.init_process_group("nccl", rank=WORLD_RANK, world_size=WORLD_SIZE)
    run()
    dist.destroy_process_group()


init_processes()
