import torch.nn.functional as F
import torch

dtype = torch.bfloat16
device = torch.device("cuda:0")

bsz, seqlen, max_seqlen = 16, 1, 256
storage_idx = torch.arange(128, 129, dtype=torch.long, device=device)
mask = torch.full((256, 256), float("-inf"), dtype=dtype, device=device)
mask = torch.triu(mask, diagonal=1)
n_heads = 32
head_dim = 128
sqrt_head_dim = 128**0.5
n_kv_heads = 8
repeats = n_heads // n_kv_heads


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return x.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def gqa_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # assumes something like
    # a.shape = (bsz, n_heads, seqlen, head_dim)
    # b.shape = (bsz, n_kv_heads, head_dim, max_seqlen)
    c = torch.empty(a.shape[:-1] + (b.shape[-1],), dtype=a.dtype, device=a.device)
    for bi in range(n_kv_heads):
        for a_step in range(repeats):
            ai = bi * repeats + a_step
            torch.matmul(
                a[:, ai : ai + 1, :, :],
                b[:, bi : bi + 1, :, :],
                out=c[:, ai : ai + 1, :, :],
            )
    return c


xq = torch.rand((bsz, n_heads, seqlen, head_dim), dtype=dtype, device=device)
keys = torch.rand((bsz, n_kv_heads, max_seqlen, head_dim), dtype=dtype, device=device)
values = torch.rand((bsz, n_kv_heads, max_seqlen, head_dim), dtype=dtype, device=device)
repeated_keys = repeat_kv(keys, repeats)
repeated_values = repeat_kv(values, repeats)

expected = F.scaled_dot_product_attention(
    xq,
    repeated_keys,
    repeated_values,
    attn_mask=mask[storage_idx],
    dropout_p=0.0,
    is_causal=False,
)

# actual = gqa_matmul(xq, keys.transpose(2, 3))
# expected = xq @ repeated_keys.transpose(2, 3)
# torch.testing.assert_close(actual, expected)

scores = gqa_matmul(xq, keys.transpose(2, 3)) / sqrt_head_dim
scores = scores + mask[storage_idx]
scores = F.softmax(scores.float(), dim=-1).type_as(xq)
test = gqa_matmul(scores, values)

torch.testing.assert_close(test, expected)
