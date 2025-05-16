import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# torch._dynamo.reset()

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return x.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

bsz = 16
seqlen = 1

@torch.compile
def forward(q, k, v):
    # repeat kv
    k = repeat_kv(k, 4)
    v = repeat_kv(v, 4)
    output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        # attn_mask=self.mask[storage_idx],
        dropout_p=0.0,
        is_causal=False,
    )
    output = output.transpose(1, 2).contiguous().reshape(bsz, seqlen, -1)
    return output

q = torch.zeros((bsz, 32, 1, 128), device="cuda", dtype=torch.bfloat16)
k = torch.zeros((bsz, 8, 256, 128), device="cuda", dtype=torch.bfloat16)
v = torch.zeros((bsz, 8, 256, 128), device="cuda", dtype=torch.bfloat16)

n_warmups = 64
n_tests = 64

# warmup
with torch.no_grad():
    for _ in range(n_warmups):
        y = forward(q, k, v)

    # test
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_tests):
        y = forward(q, k, v)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    avg_time = total_time / n_tests * 1e6
    print(f"time: {avg_time:.3f} us")
