import argparse
import json
import os
import time

import torch
import torch.distributed as dist

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])


def init_processes(max_mb):
    device = torch.device(f"cuda:{LOCAL_RANK}")
    dist.init_process_group(
        "nccl", rank=WORLD_RANK, world_size=WORLD_SIZE, device_id=device
    )
    inputs = [torch.ones((1,), dtype=torch.bfloat16, device=device)]
    batch_size = 1
    while 2 * batch_size * 1024 / 1024**2 <= max_mb:
        inputs.append(
            torch.ones((batch_size, 1024), dtype=torch.bfloat16, device=device)
        )
        batch_size *= 2

    N = 20000
    avg_latencies = []  # in ms

    for ins in inputs:
        # warmup
        for _ in range(2000):
            dist.all_reduce(ins, op=dist.ReduceOp.SUM)

        tic = time.time()
        for _ in range(N):
            dist.all_reduce(ins, op=dist.ReduceOp.SUM)
        avg_latencies.append((time.time() - tic) * 1000 / N)

    precision = 2
    if WORLD_RANK == 0:
        data = {}
        print("data_size_bytes, latency_ms, ")
        for ins, latency in zip(inputs, avg_latencies):
            ins = str(torch.numel(ins) * precision)
            latency = round(latency, 3)
            data[ins] = latency
            print(f"{ins}, {latency}, ")

        with open("inter_node_comm.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-mb", type=int, default=128)
    args = parser.parse_args()
    init_processes(args.max_mb)
    # torchrun --nnodes=2 --node-rank=0 --nproc-per-node=2 --master-addr=10.10.10.1 --master-port=9091 test_inter_node.py
