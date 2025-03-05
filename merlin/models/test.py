import torch


def precompute_freqs_cis(
    dim: int, end: int, theta: float, device: torch.device
) -> torch.Tensor:
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    print(xq_.shape)
    print(freqs_cis.shape)
    freqs_cis = freqs_cis[None, None, :, :]
    print(freqs_cis.shape)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


device = torch.device("cpu")
dtype = torch.bfloat16
bsz, seqlen, n_heads, head_dim = 16, 128, 32, 128
n_kv_heads = 8
cache = torch.empty(
    (
        2,  # key and value
        32,
        bsz,
        n_kv_heads,
        256,
        head_dim,
    ),
    dtype=dtype,
    device=device,
)
freqs_cis = precompute_freqs_cis(
    dim=head_dim,
    end=8192,
    theta=1000000.0,
    device=device,
)
storage_idx = torch.arange(128, dtype=torch.long, device=device)

xq = torch.rand((bsz, seqlen, n_heads, head_dim), dtype=dtype, device=device).transpose(1, 2)
xk = torch.rand((bsz, seqlen, n_kv_heads, head_dim), dtype=dtype, device=device).transpose(1, 2)
xq, xk = apply_rotary_emb(xq, xk, freqs_cis[storage_idx])

print(xk.shape)

# assumes bsz matches that of cache
cache[0, 0].index_copy_(dim=-2, index=storage_idx, source=xk)
# cache[1, 0].index_copy_(dim=-3, index=storage_idx, source=xv)
