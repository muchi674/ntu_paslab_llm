import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_process(rank, world_size, max_mb):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "9091"
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, device_id=device
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
    if rank == 0:
        data = {}
        print("data_size_bytes, latency_ms, ")
        for ins, latency in zip(inputs, avg_latencies):
            ins = str(torch.numel(ins) * precision)
            latency = round(latency, 3)
            data[ins] = latency
            print(f"{ins}, {latency}, ")

        with open("intra_node_comm.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-mb", type=int, default=128)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    processes = []
    mp.set_start_method("spawn")

    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, args.max_mb))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
