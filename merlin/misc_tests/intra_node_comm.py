import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def print_and_save_res(
    title: str, inputs: list[torch.Tensor], avg_latencies: list[float], filename: str
):
    data = {}
    print("-" * 20)
    print(title)
    print("-" * 20)
    print("data_size_bytes, latency_ms, ")
    for ins, latency in zip(inputs, avg_latencies):
        key = "-".join(str(d) for d in ins.shape)
        latency = round(latency, 3)
        data[key] = latency
        print(f"{key}, {latency}, ")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def init_process(
    rank: int,
    world_size: int,
    start_bsz: int,
    end_bsz: int,
    seqlen: int,
    model_dim: int,
):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "9091"
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, device_id=device
    )
    inputs = [torch.ones((1,), dtype=torch.bfloat16, device=device)]
    while start_bsz <= end_bsz:
        inputs.append(
            torch.ones(
                (start_bsz, seqlen, model_dim), dtype=torch.bfloat16, device=device
            )
        )
        start_bsz *= 2

    N = 20000
    avg_latencies = []  # in ms

    for ins in inputs:
        # warmup
        for _ in range(2000):
            dist.all_reduce(ins, op=dist.ReduceOp.MAX)

        if rank == 0:
            print(f"[collective] working on inputs with: {ins.shape}")

        tic = time.time()
        for _ in range(N):
            dist.all_reduce(ins, op=dist.ReduceOp.MAX)
        avg_latencies.append((time.time() - tic) * 1000 / N)

    if rank == 0:
        print_and_save_res(
            "INTRA-NODE COMM LATENCY", inputs, avg_latencies, "intra_coll_comm.json"
        )

    N = 4000
    avg_latencies = []  # in ms

    for ins in inputs:
        if rank > 1:
            break

        # warmup
        for _ in range(2000):
            if rank == 0:
                ops = [dist.P2POp(dist.isend, ins, 1)]
            else:
                ops = [dist.P2POp(dist.irecv, ins, 0)]
            for req in dist.batch_isend_irecv(ops):
                req.wait()

        if rank == 0:
            print(f"[P2P] working on inputs with: {ins.shape}")

        tic = time.time()

        for _ in range(N):
            if rank == 0:
                ops = [dist.P2POp(dist.isend, ins, 1)]
            else:
                ops = [dist.P2POp(dist.irecv, ins, 0)]
            for req in dist.batch_isend_irecv(ops):
                req.wait()

        avg_latencies.append((time.time() - tic) * 1000 / N)

    if rank == 0:
        print_and_save_res(
            "AVG INTRA P2P COMM LATENCY", inputs, avg_latencies, "intra_p2p_comm.json"
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-bsz", type=int, default=128)
    parser.add_argument("--end-bsz", type=int, default=128)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--model-dim", type=int)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    processes = []
    mp.set_start_method("spawn")

    for rank in range(world_size):
        p = mp.Process(
            target=init_process,
            args=(
                rank,
                world_size,
                args.start_bsz,
                args.end_bsz,
                args.seq_len,
                args.model_dim,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
