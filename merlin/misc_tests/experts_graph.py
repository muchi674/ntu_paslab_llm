from torch import nn
import torch

device = torch.device("cuda:0")
dtype = torch.bfloat16
w1 = torch.rand((14336, 4096), dtype=dtype, device=device)
w2 = torch.rand((4096, 14336), dtype=dtype, device=device)
w3 = torch.rand((14336, 4096), dtype=dtype, device=device)


def expert(x: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    x = x[start:end]
    return (nn.functional.silu(x @ w1.T) * (x @ w3.T)) @ w2.T

x = torch.ones((16, 4096), dtype=dtype, device=device)
start = torch.tensor([2], device=device)
end = torch.tensor([10], device=device)

# with torch.cuda.device(device=device):
#     graphed_expert = torch.cuda.make_graphed_callables(
#         expert,
#         (x, start, end),
#         num_warmup_iters=128,
#     )

for _ in range(1000):
    x.copy_(torch.rand((16, 4096), dtype=dtype, device=device))
    start.copy_(torch.randint(0, 8, (1,), device=device))
    end.copy_(torch.randint(8, 16, (1,), device=device))
    expert(x, start, end)
