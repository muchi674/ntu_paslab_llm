from statistics import mean
import os
import time

from torch import nn
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

world_size = 4
dtype = torch.bfloat16
model_d = 4096
interm_d = 14336
n_warmups, n_samples = 100, 10000
test_cases = [
    {
        "n_tokens": 1,
        "tp_args": (6, 2, "TP"),  # tp_size, n_activated_experts, msg,
        "tp_ep_args": (4, 2, "EP+TP"),  # on server 46
    },
    # {
    #     "n_tokens": 1,
    #     "tp_args": (6, 4, "TP"),
    #     "tp_ep_args": (2, 2, "EP+TP")
    # },
    # {
    #     "n_tokens": 1,
    #     "tp_args": (6, 6, "TP"),
    #     "tp_ep_args": (2, 3, "EP+TP")
    # },
    # {
    #     "n_tokens": 1,
    #     "tp_args": (6, 8, "TP"),
    #     "tp_ep_args": (2, 3, "EP+TP")
    # }
]


def ceildiv(a, b):
    # from: https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return -(a // -b)


def expert_forward(ws: dict, ei: int, x: torch.Tensor) -> torch.Tensor:
    w1: torch.Tensor = ws[f"{ei}.w1"].T
    w2: torch.Tensor = ws[f"{ei}.w2"]
    w3: torch.Tensor = ws[f"{ei}.w3"].T
    return (nn.functional.silu(x @ w1) * (x @ w3)) @ w2


def test(
    device: torch.device,
    tp_size: int,
    n_tokens: int,
    n_activated_experts: int,
    msg: str,
):
    adjusted_d = ceildiv(interm_d, tp_size)
    x = torch.ones((n_tokens, model_d), dtype=dtype, device=device)
    experts = {}
    for ei in range(n_activated_experts):
        experts[f"{ei}.w1"] = torch.rand(
            (adjusted_d, model_d), dtype=dtype, device=device
        )
        experts[f"{ei}.w2"] = torch.rand(
            (adjusted_d, model_d), dtype=dtype, device=device
        )
        experts[f"{ei}.w3"] = torch.rand(
            (adjusted_d, model_d), dtype=dtype, device=device
        )

    # warmup
    for _ in range(n_warmups):
        for ei in range(n_activated_experts):
            expert_forward(experts, ei, x)
    torch.cuda.synchronize(device=device)
    dist.barrier()

    latencies = []
    for _ in range(n_samples):
        tic = time.time()

        for ei in range(n_activated_experts):
            expert_forward(experts, ei, x)

        torch.cuda.synchronize(device=device)
        dist.barrier()
        latencies.append((time.time() - tic) * 1000)
    print(f"[{device}] avg {msg} latency: {round(mean(latencies), 2)} ms\n")


def init_process(rank):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "9091"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")

    for case in test_cases:
        print(case)
        n_tokens = case.pop("n_tokens")
        for args in case.values():
            tp_size, n_activated_experts, msg = args
            test(device, tp_size, n_tokens, n_activated_experts, msg)
        print()


if __name__ == "__main__":
    # assume running on 46
    processes = []
    mp.set_start_method("spawn")

    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
