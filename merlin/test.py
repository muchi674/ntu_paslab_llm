import time

import torch
import torch.nn.functional as F

device = torch.device("cuda:0")
inputs = torch.ones((2, 4096), dtype=torch.bfloat16, device=device)
gate_logits = torch.rand((2, 4096), dtype=torch.bfloat16, device=device)
weights, selected_experts = torch.topk(gate_logits, 2)
weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)

# # warmup
# for _ in range(100):
#     for ei in range(8):
#         batch_idx, nth_expert = torch.where(selected_experts == ei)
#         if torch.numel(batch_idx) == 0:
#             continue

# torch.cuda.synchronize()
# torch.cuda.cudart().cudaProfilerStart()
# torch.cuda.nvtx.range_push("test")
# tic = time.time()

# for ei in range(8):
#     batch_idx, nth_expert = torch.where(selected_experts == ei)
#     if torch.numel(batch_idx) == 0:
#         continue

# torch.cuda.synchronize()
# torch.cuda.nvtx.range_pop()
# print(time.time() - tic)
# torch.cuda.cudart().cudaProfilerStop()

# warmup
for _ in range(100):
    tmp = selected_experts.to("cpu")
    bis, nes = [], []
    for ei in range(8):
        batch_idx, nth_expert = torch.where(tmp == ei)
        if torch.numel(batch_idx) > 0:
            bis.append(batch_idx.to(device=device))
            nes.append(nth_expert.to(device=device))

torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
torch.cuda.nvtx.range_push("test")
tic = time.time()

tmp = selected_experts.to("cpu")
bis, nes = [], []
for ei in range(8):
    batch_idx, nth_expert = torch.where(tmp == ei)
    if torch.numel(batch_idx) > 0:
        bis.append(batch_idx.to(device=device))
        nes.append(nth_expert.to(device=device))

torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
print(time.time() - tic)
torch.cuda.cudart().cudaProfilerStop()
