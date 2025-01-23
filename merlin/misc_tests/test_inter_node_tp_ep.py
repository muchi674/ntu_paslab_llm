from statistics import mean
import argparse
import os
import time

from torch import nn
import torch
import torch.distributed as dist
import torch.nn.functional as F

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
        "batch_size": 1,
        "tp_args": (6, "TP"),  # tp_size, activated_experts, msg,
        "tp_ep_args": (LOCAL_WORLD_SIZE, "EP+TP"),  # both experts on 51
        "ep_args": (1, "EP"),  # both experts on 51
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
    batch_size: int,
    tp_size: int,
    msg: str,
):
    x = torch.ones((batch_size, model_d), dtype=dtype, device=GPU)
    gate_logits = torch.rand((batch_size, 8), dtype=dtype, device=GPU)
    weights, selected_experts = torch.topk(gate_logits, 2)
    weights = F.softmax(weights, dim=1, dtype=torch.float).to(dtype)

    selected_experts = selected_experts.to("cpu")
    eis, bis, nes = [], [], []
    for ei in range(8):
        batch_idx, nth_expert = torch.where(selected_experts == ei)
        if torch.numel(batch_idx) > 0:
            eis.append(ei)
            bis.append(batch_idx.to(device=GPU))
            nes.append(nth_expert.to(device=GPU))

    adjusted_d = ceildiv(interm_d, tp_size)
    if LOCAL_RANK == 0:
        print(f"expert shape: ({adjusted_d}, {model_d})")
    experts = {}
    for ei in local_experts:
        experts[f"{ei}.w1"] = torch.rand((adjusted_d, model_d), dtype=dtype, device=GPU)
        experts[f"{ei}.w2"] = torch.rand((adjusted_d, model_d), dtype=dtype, device=GPU)
        experts[f"{ei}.w3"] = torch.rand((adjusted_d, model_d), dtype=dtype, device=GPU)

    # warmup
    for _ in range(n_warmups):
        results = torch.zeros_like(x)
        for ei, batch_idx, nth_expert in zip(eis, bis, nes):
            ey = expert_forward(experts, ei, x[batch_idx])
            if ey is None:
                continue
            results[batch_idx] += weights[batch_idx, nth_expert, None] * ey
    torch.cuda.synchronize(device=GPU)
    dist.barrier()

    latencies = []
    for _ in range(n_samples):
        tic = time.time()

        results = torch.zeros_like(x)
        for ei, batch_idx, nth_expert in zip(eis, bis, nes):
            ey = expert_forward(experts, ei, x[batch_idx])
            if ey is None:
                continue
            results[batch_idx] += weights[batch_idx, nth_expert, None] * ey

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
        batch_size = case.pop("batch_size")
        for args in case.values():
            if LOCAL_RANK == 0:
                print(args)
            tp_size, msg = args
            test(expert_map[node_id], batch_size, tp_size, msg)
            if LOCAL_RANK == 0:
                print("-" * 20 + "\n")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-id", type=int)
    args = parser.parse_args()
    init_processes(args.node_id)
    # torchrun --nnodes=2 --node-rank=0 --nproc-per-node=2 --master-addr=10.10.10.1 --master-port=9090 test_inter_node_tp_ep.py --node-id 0
    # torchrun --nnodes=2 --node-rank=1 --nproc-per-node=4 --master-addr=10.10.10.1 --master-port=9090 test_inter_node_tp_ep.py --node-id 1
