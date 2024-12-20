import torch
import torch.nn.functional as F

gpu = torch.device("cuda:0")
inputs = torch.rand((4, 4096), dtype=torch.bfloat16, device=gpu)
gate_logits = torch.rand((4, 8), dtype=torch.bfloat16, device=gpu)
weights, selected_experts = torch.topk(gate_logits, 2)
weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
results = torch.zeros_like(inputs)

for _ in range(100):
    for ei in range(8):
        mask = selected_experts == ei
        flat_mask = torch.any(mask, dim=1)
        torch.sum(weights * mask, dim=1, keepdim=True)
