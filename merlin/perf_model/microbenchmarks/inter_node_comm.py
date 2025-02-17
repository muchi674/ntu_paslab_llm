import argparse
import json
import os
import time

import torch
import torch.distributed as dist

# Environment variables set by torch.distributed.launch
NODE_RANK = int(os.environ["GROUP_RANK"])
LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])


def get_global_map(device):
    global_map = torch.zeros((WORLD_SIZE, 2), dtype=torch.int64, device=device)
    local_map = torch.tensor([NODE_RANK, WORLD_RANK], dtype=torch.int64, device=device)
    dist.all_gather_into_tensor(global_map, local_map)
    return global_map


def print_and_save_res(
    title: str, inputs: list[torch.Tensor], avg_latencies: list[float], filename: str
):
    data = {}
    print("-" * 20)
    print(title)
    print("-" * 20)
    print("data_size_bytes, latency_ms, ")
    for ins, latency in zip(inputs, avg_latencies):
        ins = str(torch.numel(ins) * 2)  # bfloat16 -> 2 bytes
        latency = round(latency, 3)
        data[ins] = latency
        print(f"{ins}, {latency}, ")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def init_processes(max_mb):
    device = torch.device(f"cuda:{LOCAL_RANK}")
    dist.init_process_group(
        "nccl", rank=WORLD_RANK, world_size=WORLD_SIZE, device_id=device
    )
    global_map = get_global_map(device)
    inputs = [torch.ones((1,), dtype=torch.bfloat16, device=device)]
    batch_size = 1
    while 2 * batch_size * 1024 / 1024**2 <= max_mb:
        inputs.append(
            torch.ones((batch_size, 1024), dtype=torch.bfloat16, device=device)
        )
        batch_size *= 2

    # N = 4000
    # avg_latencies = []  # in ms

    # for ins in inputs:
    #     # warmup
    #     for _ in range(2000):
    #         dist.all_reduce(ins, op=dist.ReduceOp.MAX)

    #     tic = time.time()
    #     for _ in range(N):
    #         dist.all_reduce(ins, op=dist.ReduceOp.MAX)
    #     avg_latencies.append((time.time() - tic) * 1000 / N)

    # if WORLD_RANK == 0:
    #     print_and_save_res(
    #         "INTER COLL COMM LATENCY", inputs, avg_latencies, "inter_coll_comm.json"
    #     )

    N = 3000
    warmups = 600
    avg_latencies = []  # in ms
    receiver = 0
    sender = torch.min(
        global_map[global_map[:, 0] == 1][:, 1]
    ).item()  # first rank of the second node

    for ins in inputs:
        if WORLD_RANK == sender or WORLD_RANK == receiver:
            print(f"{WORLD_RANK} working on {torch.numel(ins) * 2}")
            # warmup
            for _ in range(warmups):
                if WORLD_RANK == sender:
                    ops = [dist.P2POp(dist.isend, ins, receiver)]
                else:
                    ops = [dist.P2POp(dist.irecv, ins, sender)]
                for req in dist.batch_isend_irecv(ops):
                    req.wait()

            tic = time.time()

            for _ in range(N):
                if WORLD_RANK == sender:
                    ops = [dist.P2POp(dist.isend, ins, receiver)]
                else:
                    ops = [dist.P2POp(dist.irecv, ins, sender)]
                for req in dist.batch_isend_irecv(ops):
                    req.wait()

            duration = time.time() - tic
            avg_latencies.append(duration * 1000 / N)
        dist.barrier()

    if WORLD_RANK == 0:
        print_and_save_res(
            "AVG INTER P2P COMM LATENCY", inputs, avg_latencies, "inter_p2p_comm.json"
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-mb", type=int, default=128)
    args = parser.parse_args()
    init_processes(args.max_mb)
    # torchrun --nnodes=2 --node-rank=0 --nproc-per-node=2 --master-addr=10.10.10.1 --master-port=9091 test_inter_node.py
