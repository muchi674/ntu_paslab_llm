import time
import nvtx
import torch
import torch.nn.functional as F


### def forward(self, inputs: torch.Tensor) -> torch.Tensor:

device = torch.device("cuda:0")
inputs = torch.ones((2, 4096), dtype=torch.bfloat16, device=device)
gate_logits = torch.rand((4096, 8), dtype=torch.bfloat16, device=device)
print(gate_logits)
weights, selected_experts = torch.topk(gate_logits, 2)
print(weights)
print(selected_experts)
weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)


# warmup
# for _ in range(100):
#     tmp = selected_experts.to("cpu")
#     bis, nes = [], []
#     for ei in range(8):
#         batch_idx, nth_expert = torch.where(tmp == ei)
#         if torch.numel(batch_idx) > 0:
#             bis.append(batch_idx.to(device=device))
#             nes.append(nth_expert.to(device=device))

# torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
torch.cuda.nvtx.range_push("test")
tic = time.time()

tmp = selected_experts.to("cpu")
eis, bis, nes = [], [], []
### original
with nvtx.annotate("data transferr", color="purple"):
    for ei in range(8):
        batch_idx, nth_expert = torch.where(tmp == ei)
        if torch.numel(batch_idx) > 0:
            eis.append(ei)
            bis.append(batch_idx.to(device=device))
            nes.append(nth_expert.to(device=device))

### one memcpy
# select_shape = []   # for one memcpy version, save the tensor's size
# with nvtx.annotate("data transferr", color="purple"):

#     for ei in range(8):
#         batch_idx, nth_expert = torch.where(tmp == ei)
#         if torch.numel(batch_idx) > 0:
#             eis.append(ei)
#             bis.append(batch_idx)
#             nes.append(nth_expert)
#             select_shape.append(len(batch_idx))
#     concat_bis = torch.cat(bis, dim=0)
#     concat_bis_cpu = concat_bis.cpu()
#     bis = torch.split(concat_bis_cpu, [len(t) for t in bis])

#     concat_nes = torch.cat(bis, dim=0)
#     concat_nes_cpu = concat_nes.cpu()
#     nes = torch.split(concat_nes_cpu, [len(t) for t in nes])
#     # bis = bis.to(device=device)
#     # nes = nes.to(device=device)

torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
print(time.time() - tic)
torch.cuda.cudart().cudaProfilerStop()

