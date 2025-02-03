import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])

def get_node_group(node_id, gpu, global_group):
    global_map = torch.zeros((WORLD_SIZE, 2), dtype=torch.int64, device=gpu)
    local_map = torch.tensor([node_id, WORLD_RANK], dtype=torch.int64, device=gpu)
    dist.all_gather_into_tensor(global_map, local_map, group=global_group)
    ranks_on_node = global_map[global_map[:, 0] == node_id][:, 1]
    return dist.new_group(ranks_on_node.tolist(), use_local_synchronization=True)

def init_processes(node_id):
    gpu = torch.device(f"cuda:{LOCAL_RANK}")
    dist.init_process_group(
        "nccl", rank=WORLD_RANK, world_size=WORLD_SIZE, device_id=gpu
    )
    global_group = dist.new_group(
        list(range(WORLD_SIZE)), use_local_synchronization=True
    )
    node_group = get_node_group(node_id, gpu, global_group)

    x = torch.ones((1, 4096), dtype=torch.bfloat16, device=gpu)
    # warmup
    for _ in range(1000):
        dist.all_reduce(x, op=dist.ReduceOp.MAX, group=node_group)
        dist.all_reduce(x, op=dist.ReduceOp.MAX, group=global_group)

    dist.barrier(group=global_group)
    tic = time.time()

    for _ in range(10000):
        dist.all_reduce(x, op=dist.ReduceOp.MAX, group=node_group)
        dist.all_reduce(x, op=dist.ReduceOp.MAX, group=global_group)

    print(f"AVG run latency: {((time.time() - tic) * 1000) / 10000} ms")
    dist.barrier(group=global_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-id", type=int)
    args = parser.parse_args()
    init_processes(args.node_id)
    # torchrun --nnodes=2 --node-rank=0 --nproc-per-node=2 --master-addr=10.10.10.1 --master-port=9091 test_inter_node.py
