from statistics import mean
import argparse
import os
import time

from torch import nn
import torch
import torch.distributed as dist

# Environment variables set by torch.distributed.launch
LOCAL_WORLD_SIZE = torch.cuda.device_count()
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])
GPU = torch.device(f"cuda:{LOCAL_RANK}")

dtype = torch.bfloat16
model_d = 4096
interm_d = 14336
n_warmups, n_samples = 100, 10000
expert_map = {0: (0, 1, 2), 1: (3, 4, 5, 6, 7)}  # node_id: experts responsible
test_cases = [
    {
        "n_tokens": 1,
        "tp_args": (6, [0, 1], "TP"),  # tp_size, activated_experts, msg,
        "tp_ep_args": (LOCAL_WORLD_SIZE, [0, 3], "EP+TP"),  # both experts on 51
        "ep_args": (1, [0, 3], "EP"),  # both experts on 51
    }
]


def ceildiv(a, b):
    # from: https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return -(a // -b)


def expert_forward(ws: dict, ei: int, x: torch.Tensor) -> torch.Tensor:
    if f"{ei}.w1" not in ws:
        return None
    w1: torch.Tensor = ws[f"{ei}.w1"].T
    w2: torch.Tensor = ws[f"{ei}.w2"]
    w3: torch.Tensor = ws[f"{ei}.w3"].T
    return (nn.functional.silu(x @ w1) * (x @ w3)) @ w2


def test(
    local_experts: tuple[int],
    n_tokens: int,
    tp_size: int,
    activated_experts: tuple[int],
    msg: str,
):
    adjusted_d = ceildiv(interm_d, tp_size)
    x = torch.ones((n_tokens, model_d), dtype=dtype, device=GPU)
    if LOCAL_RANK == 0:
        print(f"expert shape: ({adjusted_d}, {model_d})")
    experts = {}
    for ei in local_experts:
        experts[f"{ei}.w1"] = torch.rand((adjusted_d, model_d), dtype=dtype, device=GPU)
        experts[f"{ei}.w2"] = torch.rand((adjusted_d, model_d), dtype=dtype, device=GPU)
        experts[f"{ei}.w3"] = torch.rand((adjusted_d, model_d), dtype=dtype, device=GPU)

    # warmup
    for _ in range(n_warmups):
        for ei in activated_experts:
            expert_forward(experts, ei, x)
    torch.cuda.synchronize(device=GPU)
    dist.barrier()

    latencies = []
    for _ in range(n_samples):
        tic = time.time()

        for ei in activated_experts:
            expert_forward(experts, ei, x)

        torch.cuda.synchronize(device=GPU)
        dist.barrier()
        latencies.append((time.time() - tic) * 1000)
    if LOCAL_RANK == 0:
        print(f"avg {msg} latency: {round(mean(latencies), 2)} ms")


def init_processes(node_id):
    dist.init_process_group(
        "nccl", rank=WORLD_RANK, world_size=WORLD_SIZE, device_id=GPU
    )

    for case in test_cases:
        n_tokens = case.pop("n_tokens")
        for args in case.values():
            if LOCAL_RANK == 0:
                print(args)
            tp_size, activated_experts, msg = args
            test(expert_map[node_id], n_tokens, tp_size, activated_experts, msg)
            if LOCAL_RANK == 0:
                print("-" * 20 + "\n")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-id", type=int)
    args = parser.parse_args()
    init_processes(args.node_id)
    # torchrun --nnodes=2 --node-rank=0 --nproc-per-node=2 --master-addr=10.10.10.1 --master-port=9091 test_inter_node_tp_ep.py --node-id 0
    # torchrun --nnodes=2 --node-rank=1 --nproc-per-node=4 --master-addr=10.10.10.1 --master-port=9091 test_inter_node_tp_ep.py --node-id 1
