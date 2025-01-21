from statistics import mean
import time

from torch import nn
import torch


def ceildiv(a, b):
    # from: https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return -(a // -b)


def expert_forward(ws: dict, ei: int, x: torch.Tensor) -> torch.Tensor:
    w1: torch.Tensor = ws[f"{ei}.w1"].T
    w2: torch.Tensor = ws[f"{ei}.w2"]
    w3: torch.Tensor = ws[f"{ei}.w3"].T
    return (nn.functional.silu(x @ w1) * (x @ w3)) @ w2


dtype = torch.bfloat16
device = torch.device("cuda:0")
model_d = 4096
interm_d = 14336
batch_sizes = [1, 4, 16, 64, 128]
n_warmups, n_samples = 100, 10000

B = 2

# TP
n_activated_experts = 8
tp_size = 6
adjusted_d = ceildiv(interm_d, tp_size)
x = torch.ones((B, model_d), dtype=dtype, device=device)
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

tp_latencies = []
for _ in range(n_samples):
    tic = time.time()

    for ei in range(n_activated_experts):
        expert_forward(experts, ei, x)

    torch.cuda.synchronize(device=device)
    tp_latencies.append((time.time() - tic) * 1000)
print(f"avg TP latency: {round(mean(tp_latencies), 2)} ms")

# TP + EP, on server 51
n_activated_experts = 3
tp_size = 2
adjusted_d = ceildiv(interm_d, tp_size)
x = torch.ones((B, model_d), dtype=dtype, device=device)
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

ep_tp_latencies = []
for _ in range(n_samples):
    tic = time.time()

    for ei in range(n_activated_experts):
        expert_forward(experts, ei, x)

    torch.cuda.synchronize(device=device)
    ep_tp_latencies.append((time.time() - tic) * 1000)
print(f"avg EP+TP latency: {round(mean(ep_tp_latencies), 2)} ms")
