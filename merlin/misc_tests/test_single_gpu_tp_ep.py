from statistics import mean
import time

from torch import nn
import torch

dtype = torch.bfloat16
device = torch.device("cuda:0")
model_d = 4096
interm_d = 14336
n_warmups, n_samples = 100, 10000


def ceildiv(a, b):
    # from: https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return -(a // -b)


def expert_forward(ws: dict, ei: int, x: torch.Tensor) -> torch.Tensor:
    w1: torch.Tensor = ws[f"{ei}.w1"].T
    w2: torch.Tensor = ws[f"{ei}.w2"]
    w3: torch.Tensor = ws[f"{ei}.w3"].T
    return (nn.functional.silu(x @ w1) * (x @ w3)) @ w2


def test(tp_size: int, n_tokens: int, n_activated_experts: int, msg: str):
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

    latencies = []
    for _ in range(n_samples):
        tic = time.time()

        for ei in range(n_activated_experts):
            expert_forward(experts, ei, x)

        torch.cuda.synchronize(device=device)
        latencies.append((time.time() - tic) * 1000)
    print(f"avg {msg} latency: {round(mean(latencies), 2)} ms")

TEST_CASES = [
    {
        "n_tokens": 1,
        "tp_args": (6, 2, "TP"), # tp_size, n_activated_experts, msg,
        "tp_ep_args": (4, 2, "EP+TP") # on node 51, worst case
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

for case in TEST_CASES:
    print(case)
    n_tokens = case.pop("n_tokens")
    for args in case.values():
        tp_size, n_activated_experts, msg = args
        test(tp_size, n_tokens, n_activated_experts, msg)
    print()
