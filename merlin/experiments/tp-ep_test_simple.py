import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import time

torch.manual_seed(5678)

@dataclass
class ModelArgs:
    dim = 4096
    hidden_dim = 14336
    num_experts = 8
    num_experts_per_tok = 2


class Experts:
    def __init__(self, ws: dict):
        self.ws: dict[str, torch.Tensor] = ws

    def forward(self, ei: int, x: torch.Tensor) -> torch.Tensor:
        if f"{ei}.w1" not in self.ws:
            return None
        
        w1: torch.Tensor = self.ws[f"{ei}.w1"].T
        w2: torch.Tensor = self.ws[f"{ei}.w2"]
        w3: torch.Tensor = self.ws[f"{ei}.w3"].T

        return (nn.functional.silu(x @ w1) * (x @ w3)) @ w2
    
class MoeLayer(nn.Module):
    def __init__(self, args: ModelArgs, experts: Experts):
        super().__init__()
        self.num_experts: int = args.num_experts
        self.num_experts_per_tok: int = args.num_experts_per_tok
        self.gate = nn.Linear(args.dim, args.num_experts, bias=False)
        self.experts = experts
        self.comp_time = []

    def forward(self, inputs: torch.Tensor, need_profile: bool) -> torch.Tensor:
        # computation       
        results = torch.zeros_like(inputs)

        inputs = torch.rand(
            (inputs.shape[0]//4, 4096),
            dtype=inputs.dtype,
            device=inputs.device
        )
        torch.cuda.synchronize(device=inputs.device)
        tic = time.time()
        for ei in range(self.num_experts):
            # ey = self.experts.forward(ei, inputs[batch_idx])
            ey = self.experts.forward(ei, inputs)
            if ey is None:
                continue
        torch.cuda.synchronize(device=inputs.device)
        toc = time.time()

        torch.cuda.synchronize(device=inputs.device)
        if need_profile:
           self.comp_time.append((toc-tic)*1000)

        return results


device = "cuda:0"
args = ModelArgs()

# =========================== #
experts_on_node = [i for i in range(0, 8)]
tp_size = 6
batch_size = 32
# =========================== #

# initialize expert weights
expert_weights = {}
for ei in range(args.num_experts):
    if ei in experts_on_node:
        expert_weights[f"{ei}.w1"] = torch.rand(
            args.hidden_dim//tp_size,
            args.dim,
            dtype = torch.bfloat16,
            device=device
        )
        expert_weights[f"{ei}.w2"] = torch.rand(
            args.hidden_dim//tp_size,
            args.dim,
            dtype = torch.bfloat16,
            device=device
        )
        expert_weights[f"{ei}.w3"] = torch.rand(
            args.hidden_dim//tp_size,
            args.dim,
            dtype = torch.bfloat16,
            device=device
        )
        
experts = Experts(expert_weights)
model = MoeLayer(args, experts)
model.to(device)

num_iter = 200
with torch.no_grad():
    # warm up
    for _ in range(100):
        x = torch.rand((batch_size, args.dim), dtype=torch.bfloat16, device=device)
        y = model(x, need_profile=False)
    
    for _ in range(num_iter):
        x = torch.rand((batch_size, args.dim), dtype=torch.bfloat16, device=device)
        y = model(x, need_profile=True)
    
    #print(y)
    print(f"{sum(model.comp_time) / num_iter:.3f}")
    # print(count)
