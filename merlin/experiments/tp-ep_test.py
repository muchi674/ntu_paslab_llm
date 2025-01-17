import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import time

torch.manual_seed(1234)

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
    
count = [0 for _ in range(8)]
class MoeLayer(nn.Module):
    def __init__(self, args: ModelArgs, experts: Experts):
        super().__init__()
        self.num_experts: int = args.num_experts
        self.num_experts_per_tok: int = args.num_experts_per_tok
        self.gate = nn.Linear(args.dim, args.num_experts, bias=False)
        self.experts = experts
        self.comp_time = []
        self.comm_time = []

    def forward(self, inputs: torch.Tensor, need_profile: bool) -> torch.Tensor:
        # computation
        # gate_logits = self.gate(inputs)
        # weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        # weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        
        weights = torch.rand(
            (inputs.shape[0], self.num_experts_per_tok),
            dtype=inputs.dtype,
            device=inputs.device
        )

        probs = torch.ones((inputs.shape[0], self.num_experts))
        # probs = torch.tensor([[0.4, 0.4, 0.2, 0, 0, 0, 0, 0] for _ in range(inputs.shape[0])], dtype=torch.bfloat16)
        selected_experts = torch.multinomial(probs, self.num_experts_per_tok)
        # selected_experts = torch.randint(
        #     0, self.num_experts,
        #     (inputs.shape[0], self.num_experts_per_tok),
        #     device=inputs.device
        # )
        results = torch.zeros_like(inputs)
        
        selected_experts = selected_experts.to("cpu")
        eis, bis, nes = [], [], []
        for ei in range(self.num_experts):
            batch_idx, nth_expert = torch.where(selected_experts == ei)
            if torch.numel(batch_idx) > 0:
                eis.append(ei)
                bis.append(batch_idx.to(device=inputs.device))
                nes.append(nth_expert.to(device=inputs.device))
        
        # count tokens for each expert 
        # for i in range(selected_experts.shape[0]):
        #     for j in range(selected_experts.shape[1]):
        #         count[selected_experts[i][j].item()] += 1

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for ei, batch_idx, nth_expert in zip(eis, bis, nes):
            ey = self.experts.forward(ei, inputs[batch_idx])
            if ey is None:
                continue
            results[batch_idx] += weights[batch_idx, nth_expert, None] * ey
        end.record()

        torch.cuda.synchronize(device=inputs.device)
        if need_profile:
           self.comp_time.append(start.elapsed_time(end))

        return results


device = "cuda:0"
args = ModelArgs()

# =========================== #
experts_on_node = [i for i in range(3, 8)]
tp_size = 4
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
    print(sum(model.comp_time) / num_iter)
    # print(count)
