import torch

torch.manual_seed(0)
device = torch.device("cuda:0")
max_batch_size = 2
max_seq_len = 128
seqlen_sum = max_batch_size * max_seq_len
n_kv_heads, head_dim = 8, 128


def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


# TODO: this needs fixing
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


theta = 1000000.0
freqs_cis = precompute_freqs_cis(head_dim, 128_000, theta)[0:max_seq_len]
xq = torch.rand(
    (max_batch_size, max_seq_len, n_kv_heads, head_dim),
    dtype=torch.bfloat16,
    device=device,
)
xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
expected = reshape_for_broadcast(freqs_cis, xq_)
test = freqs_cis[None,:,None,:]
print(torch.equal(expected, test))
