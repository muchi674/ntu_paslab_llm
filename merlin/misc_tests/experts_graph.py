import time

from torch import nn
import torch

device = torch.device("cuda:0")
dtype = torch.bfloat16
N = 1000
w1s = [torch.rand((14336, 4096), dtype=dtype, device=device) for _ in range(N)]
w2s = [torch.rand((4096, 14336), dtype=dtype, device=device) for _ in range(N)]
w3s = [torch.rand((14336, 4096), dtype=dtype, device=device) for _ in range(N)]
xs = [torch.ones((16, 4096), dtype=dtype, device=device) for _ in range(N)]


def expert(x: torch.Tensor) -> torch.Tensor:
    return (nn.functional.silu(x @ w1s[0].T) * (x @ w3s[0].T)) @ w2s[0].T


def weightless_expert(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor
) -> torch.Tensor:
    return (nn.functional.silu(x @ w1.T) * (x @ w3.T)) @ w2.T


with torch.cuda.device(device=device):
    graphed_expert = torch.cuda.make_graphed_callables(
        expert,
        (xs[0],),
        num_warmup_iters=128,
    )


# warmup
for i in range(1000):
    expert(xs[i])
torch.cuda.synchronize(device=device)

tic = time.time()
for i in range(1000):
    expert(xs[i])
torch.cuda.synchronize(device=device)
print(f"took {(time.time() - tic) / 1000} sec per op")

with torch.cuda.device(device=device):
    graphed_weightless_expert = torch.cuda.make_graphed_callables(
        weightless_expert,
        (xs[0], w1s[0], w2s[0], w3s[0]),
        num_warmup_iters=128,
    )

# warmup
for _ in range(1000):
    graphed_weightless_expert(xs[i], w1s[i], w2s[i], w3s[i])
torch.cuda.synchronize(device=device)

tic = time.time()
for _ in range(1000):
    graphed_weightless_expert(xs[i], w1s[i], w2s[i], w3s[i])
torch.cuda.synchronize(device=device)
print(f"took {(time.time() - tic) / 1000} sec per op")
