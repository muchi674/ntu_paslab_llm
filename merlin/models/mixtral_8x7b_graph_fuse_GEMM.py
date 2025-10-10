import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from torch import nn

import triton
import triton.language as tl

from math import ceil

# Environment variables set by torch.distributed.launch
LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
GROUP_RANK = int(os.environ["GROUP_RANK"])
WORLD_RANK = int(os.environ["RANK"])

DEFAULT_SEED = 7


def precompute_rope_cos_sin(dim: int, end: int, theta: float, device):
    
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
    )  # [D/2]
    t = torch.arange(end, device=device, dtype=torch.float32)  # [T]
    ang = torch.outer(t, freqs).float()  # [T, D/2]
    cos, sin = ang.cos(), ang.sin()
    return cos, sin



@triton.jit
def rope_fused_kernel(
    Q_ptr, K_ptr,        # [B, Hq/Hk, T, D]
    OQ_ptr, OK_ptr,      # 输出
    COS_ptr, SIN_ptr,    # [T, D/2]
    B, Hq, Hk, T, D, Hmax,
    stride_q_b, stride_q_h, stride_q_t, stride_q_d,
    stride_k_b, stride_k_h, stride_k_t, stride_k_d,
    stride_oq_b, stride_oq_h, stride_oq_t, stride_oq_d,
    stride_ok_b, stride_ok_h, stride_ok_t, stride_ok_d,
    stride_cs_t, stride_cs_dh,
    BLOCK_D: tl.constexpr,
):
    # program ids
    bh   = tl.program_id(0)      # 0 .. B*Hmax-1
    t_ix = tl.program_id(1)      # 0 .. T-1
    db   = tl.program_id(2)      # D block

    h_ix = bh % Hmax
    b_ix = bh // Hmax

    d_start = db * BLOCK_D
    offs_d  = d_start + tl.arange(0, BLOCK_D)
    mask_d  = offs_d < D

    # pair index + even/odd
    i_pair   = offs_d // 2
    is_even  = (offs_d % 2) == 0
    mask_pair = (i_pair < (D // 2)) & mask_d

    # active heads
    active_q = h_ix < Hq
    active_k = h_ix < Hk

    # cos/sin[t, i_pair]
    cos_ptr = COS_ptr + t_ix * stride_cs_t + i_pair * stride_cs_dh
    sin_ptr = SIN_ptr + t_ix * stride_cs_t + i_pair * stride_cs_dh
    cos = tl.load(cos_ptr, mask=mask_pair, other=1.0).to(tl.float32)
    sin = tl.load(sin_ptr, mask=mask_pair, other=0.0).to(tl.float32)

    # ---- Q path (masked) ----
    q_row = Q_ptr  + b_ix*stride_q_b  + h_ix*stride_q_h  + t_ix*stride_q_t
    oq_row= OQ_ptr + b_ix*stride_oq_b + h_ix*stride_oq_h + t_ix*stride_oq_t
    mask_q = mask_pair & active_q
    q_even = tl.load(q_row + (i_pair*2)   * stride_q_d, mask=mask_q, other=0.0).to(tl.float32)
    q_odd  = tl.load(q_row + (i_pair*2+1) * stride_q_d, mask=mask_q & (offs_d+1 < D), other=0.0).to(tl.float32)
    q_even_p = q_even * cos - q_odd * sin
    q_odd_p  = q_even * sin + q_odd * cos
    q_out    = tl.where(is_even, q_even_p, q_odd_p)
    tl.store(oq_row + offs_d * stride_oq_d, q_out, mask=mask_q)

    # ---- K path (masked) ----
    k_row = K_ptr  + b_ix*stride_k_b  + h_ix*stride_k_h  + t_ix*stride_k_t
    ok_row= OK_ptr + b_ix*stride_ok_b + h_ix*stride_ok_h + t_ix*stride_ok_t
    mask_k = mask_pair & active_k
    k_even = tl.load(k_row + (i_pair*2)   * stride_k_d, mask=mask_k, other=0.0).to(tl.float32)
    k_odd  = tl.load(k_row + (i_pair*2+1) * stride_k_d, mask=mask_k & (offs_d+1 < D), other=0.0).to(tl.float32)
    k_even_p = k_even * cos - k_odd * sin
    k_odd_p  = k_even * sin + k_odd * cos
    k_out    = tl.where(is_even, k_even_p, k_odd_p)
    tl.store(ok_row + offs_d * stride_ok_d, k_out, mask=mask_k)


def rope_fused(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_d: int = 128):
    assert xq.ndim == 4 and xk.ndim == 4
    B, Hq, T, D  = xq.shape
    Bk, Hk, Tk, Dk = xk.shape
    assert B == Bk and T == Tk and D == Dk
    assert D % 2 == 0
    assert cos.shape == (T, D // 2) and sin.shape == (T, D // 2)

    xq = xq.contiguous()
    xk = xk.contiguous()
    oq = torch.empty_like(xq)
    ok = torch.empty_like(xk)

    sq_b, sq_h, sq_t, sq_d   = xq.stride()
    sk_b, sk_h, sk_t, sk_d   = xk.stride()
    soq_b, soq_h, soq_t, soq_d = oq.stride()
    sok_b, sok_h, sok_t, sok_d = ok.stride()
    sc_t, sc_dh = cos.stride()

    Hmax = max(Hq, Hk)
    grid = (B * Hmax, T, ceil(D / block_d))

    rope_fused_kernel[grid](
        xq, xk, oq, ok, cos, sin,
        B, Hq, Hk, T, D, Hmax,
        sq_b, sq_h, sq_t, sq_d,
        sk_b, sk_h, sk_t, sk_d,
        soq_b, soq_h, soq_t, soq_d,
        sok_b, sok_h, sok_t, sok_d,
        sc_t, sc_dh,
        BLOCK_D=block_d,
        num_warps=4 if block_d <= 128 else 8,
        num_stages=2,
    )
    return oq, ok


################################
# 下面是 RMSNorm 的 kernel fuse
################################
@triton.jit
def rmsnorm_fused_kernel(
    X_ptr,            
    W_ptr,            
    Y_ptr,            
    eps,              
    M, D,             
    stride_xm, stride_xd,
    stride_ym, stride_yd,
    stride_wd,
    inv_D,
    BLOCK_D: tl.constexpr,
):
    m = tl.program_id(0)              # 0..M-1
    x_row = X_ptr + m * stride_xm
    y_row = Y_ptr + m * stride_ym

    # --- pass 1: sum of squares along D ---
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for off in range(0, D, BLOCK_D):
        d = off + tl.arange(0, BLOCK_D)
        mask = d < D
        x = tl.load(x_row + d * stride_xd, mask=mask, other=0.0)
        x = x.to(tl.float32)
        acc += x * x
    ss = tl.sum(acc, axis=0)
    mean = ss * inv_D 
    inv_rms = tl.math.rsqrt(mean + eps)

    # --- pass 2: write normalized * weight ---
    for off in range(0, D, BLOCK_D):
        d = off + tl.arange(0, BLOCK_D)
        mask = d < D
        x = tl.load(x_row + d * stride_xd, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + d * stride_wd, mask=mask, other=1.0).to(tl.float32)
        y = x * inv_rms * w
        tl.store(y_row + d * stride_yd, y.to(Y_ptr.dtype.element_ty), mask=mask)


def rmsnorm_fused(x: torch.Tensor, weight: torch.Tensor, eps: float, block_d: int = 256):
    assert x.is_cuda, "RMSNorm input must be on CUDA"
    assert weight.is_cuda, "RMSNorm weight must be on CUDA"
    assert x.shape[-1] == weight.shape[0], "RMSNorm: weight size must equal last dim"

    D = x.shape[-1]
    M = x.numel() // D

    dev = x.device
    x_2d = x.contiguous().view(M, D)
    w = weight.contiguous()
    y_2d = torch.empty_like(x_2d)

    sx_m, sx_d = x_2d.stride()
    sy_m, sy_d = y_2d.stride()
    (sw_d,) = w.stride()

    grid = (M,)
    inv_D = float(1.0 / D)

    with torch.cuda.device(dev):
        rmsnorm_fused_kernel[grid](
            x_2d, w, y_2d,
            eps,
            M, D,
            sx_m, sx_d,
            sy_m, sy_d,
            sw_d,
            inv_D,                   
            BLOCK_D=block_d,
            num_warps=4 if block_d <= 256 else 8,
            num_stages=2,
        )
    return y_2d.view_as(x)


################################
# 下面是 repeat_kv 的 kernel fuse
################################
@triton.jit
def repeat_kv_fused_kernel(
    K_in, V_in,           # [B, Hk, T, D]
    K_out, V_out,         # [B, Hq, T, D]，Hq = Hk * n_rep
    B, Hk, Hq, T, D, n_rep,
    stride_ki_b, stride_ki_h, stride_ki_t, stride_ki_d,
    stride_vi_b, stride_vi_h, stride_vi_t, stride_vi_d,
    stride_ko_b, stride_ko_h, stride_ko_t, stride_ko_d,
    stride_vo_b, stride_vo_h, stride_vo_t, stride_vo_d,
    BLOCK_D: tl.constexpr,
):
    # grid: (B*Hq, T, ceil(D/BLOCK_D))
    bh   = tl.program_id(0)     # 0 .. B*Hq-1
    t_ix = tl.program_id(1)     # 0 .. T-1
    db   = tl.program_id(2)     # D block

    hq_ix = bh % Hq            
    b_ix  = bh // Hq
    hk_ix = hq_ix // n_rep

    d_start = db * BLOCK_D
    offs_d  = d_start + tl.arange(0, BLOCK_D)
    m_d     = offs_d < D

    k_src = K_in + b_ix*stride_ki_b + hk_ix*stride_ki_h + t_ix*stride_ki_t
    v_src = V_in + b_ix*stride_vi_b + hk_ix*stride_vi_h + t_ix*stride_vi_t
    k_dst = K_out + b_ix*stride_ko_b + hq_ix*stride_ko_h + t_ix*stride_ko_t
    v_dst = V_out + b_ix*stride_vo_b + hq_ix*stride_vo_h + t_ix*stride_vo_t

    k_val = tl.load(k_src + offs_d*stride_ki_d, mask=m_d, other=0.0)
    v_val = tl.load(v_src + offs_d*stride_vi_d, mask=m_d, other=0.0)

    tl.store(k_dst + offs_d*stride_ko_d, k_val, mask=m_d)
    tl.store(v_dst + offs_d*stride_vo_d, v_val, mask=m_d)


def repeat_kv_fused(keys: torch.Tensor,
                    values: torch.Tensor,
                    n_rep: int,
                    block_d: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert keys.ndim == 4 and values.ndim == 4 and keys.shape == values.shape
    B, Hk, T, D = keys.shape
    if n_rep == 1:
        return keys, values
    Hq = Hk * n_rep

    k_in = keys.contiguous()
    v_in = values.contiguous()
    k_out = torch.empty((B, Hq, T, D), device=k_in.device, dtype=k_in.dtype)
    v_out = torch.empty((B, Hq, T, D), device=v_in.device, dtype=v_in.dtype)

    ski_b, ski_h, ski_t, ski_d = k_in.stride()
    svi_b, svi_h, svi_t, svi_d = v_in.stride()
    sko_b, sko_h, sko_t, sko_d = k_out.stride()
    svo_b, svo_h, svo_t, svo_d = v_out.stride()

    grid = (B * Hq, T, (D + block_d - 1) // block_d)
    repeat_kv_fused_kernel[grid](
        k_in, v_in, k_out, v_out,
        B, Hk, Hq, T, D, n_rep,
        ski_b, ski_h, ski_t, ski_d,
        svi_b, svi_h, svi_t, svi_d,
        sko_b, sko_h, sko_t, sko_d,
        svo_b, svo_h, svo_t, svo_d,
        BLOCK_D=block_d,
        num_warps=4 if block_d <= 128 else 8,
        num_stages=2,
    )
    return k_out, v_out


@triton.jit
def _topk_softmax_smallE_kernel(
    logits_ptr,          # [M, E]
    out_val_ptr,         # [M, K]
    out_idx_ptr,         # [M, K] (int64)
    M, E, K,
    stride_lm, stride_le,
    stride_om, stride_ok,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)           # one program per row
    row = pid
    # bounds
    if row >= M:
        return

    # ---- 读一行 logits，并转成 fp32 计算 ----
    offs_e = tl.arange(0, BLOCK_E)
    vals = tl.zeros([BLOCK_E], dtype=tl.float32)
    idxs = tl.zeros([BLOCK_E], dtype=tl.int32)
    # 按块加载；E 往往很小（8/16/32/64），通常一次就读完
    acc = tl.zeros([E], dtype=tl.float32)  # 备用：若 E > BLOCK_E，可以改为流式读；此处简单实现
    # 读取全行
    for start in range(0, E, BLOCK_E):
        mask = (offs_e + start) < E
        v = tl.load(
            logits_ptr + row * stride_lm + (offs_e + start) * stride_le,
            mask=mask,
            other=-float("inf"),
        )
        v = v.to(tl.float32)
        # 写进 acc
        # 注意：acc 是长度为 E 的一维向量；用逐元素写
        # 这里用 scatter，Triton 当前不支持直接 variable indices scatter，只能分支处理
        # 简化：BLOCK_E 取 >= E，避免多段；否则可改成 while + tl.store 子片段。
        if start == 0:
            vals = v
            idxs = (offs_e + start).to(tl.int32)
        else:
            # 合并到 vals/idxs：这里为了简单，假设 E <= BLOCK_E（Mixtral 的 E=8 满足）
            pass

    # 如果 E <= BLOCK_E，vals/idxs 即整行
    # ---- 线性扫描维护 K 个槽位（降序）----
    # 初始为 -inf 和 -1
    topv = tl.full([16], -float("inf"), tl.float32)   # 支持 K<=16
    topi = tl.full([16], -1, tl.int32)

    # 只用到前 K 个槽位，后面不访问
    # 逐元素扫描 vals
    for j in range(0, E):
        v = vals[j]
        i = idxs[j]
        # 插入到有序 top-K（降序）
        # 与槽位比较，从大到小“冒泡”插入
        pos = K - 1
        # 如果比最后一名还小，直接丢弃
        if v <= topv[pos]:
            # 但如果初始是 -inf，会进入下面循环
            # 用 while 将 v 插入到合适位置
            pass
        # 向上挪动
        while (pos > 0) & (v > topv[pos - 1]):
            topv[pos] = topv[pos - 1]
            topi[pos] = topi[pos - 1]
            pos = pos - 1
        # 放入最终位置
        if v > topv[pos]:
            topv[pos] = v
            topi[pos] = i
        else:
            # 如果 v <= topv[pos] 但该位置还没填（-inf），也要写入
            if topi[pos] < 0:
                topv[pos] = v
                topi[pos] = i

    # ---- 对 top-K 做 softmax（fp32 计算）----
    # 只取前 K 项
    # 减最大值防溢出
    maxv = -float("inf")
    for t in range(0, K):
        maxv = tl.maximum(maxv, topv[t])
    sumexp = 0.0
    for t in range(0, K):
        sumexp += tl.exp(topv[t] - maxv)
    # 归一化
    for t in range(0, K):
        sm = tl.exp(topv[t] - maxv) / sumexp
        # 写回
        tl.store(out_val_ptr + row * stride_om + t * stride_ok, sm)
        tl.store(out_idx_ptr + row * stride_om + t * stride_ok, topi[t].to(tl.int64))


def triton_topk_softmax_smallE(
    gate_logits: torch.Tensor, k: int, block_e: int = 64
):
    """
    gate_logits: [M, E] (bf16/fp16/fp32), E 小(例如 8/16/32/64)
    返回: (topk_vals [M,k] same dtype as input, topk_idx [M,k] int64)
    """
    assert gate_logits.ndim == 2
    M, E = gate_logits.shape
    assert k <= 16, "K too large for this simple kernel"
    assert E <= block_e, "For simplicity, set BLOCK_E >= E (e.g., 64 when E<=64)"

    gl = gate_logits.contiguous()
    # 输出：用 fp32 计算，再转回输入 dtype
    out_val = torch.empty((M, k), device=gl.device, dtype=torch.float32)
    out_idx = torch.empty((M, k), device=gl.device, dtype=torch.int64)

    grid = (M,)
    _topk_softmax_smallE_kernel[grid](
        gl, out_val, out_idx,
        M, E, k,
        gl.stride(0), gl.stride(1),
        out_val.stride(0), out_val.stride(1),
        BLOCK_E=block_e,
        num_warps=2,
        num_stages=1,
    )
    return out_val.to(gl.dtype), out_idx

def get_json(file_path: Path) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    # assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float
    moe: dict
    first_layer: int = None
    last_layer: int = None
    has_pp: bool = False
    parallel_experts: bool = False
    inter_parallel_attn: bool = False
    intra_parallel_attn: bool = False

    @classmethod
    def from_hf_config(cls, params: dict):
        return cls(
            dim=params["hidden_size"],
            n_layers=params["num_hidden_layers"],
            head_dim=params["hidden_size"] // params["num_attention_heads"],
            hidden_dim=params["intermediate_size"],
            n_heads=params["num_attention_heads"],
            n_kv_heads=params["num_key_value_heads"],
            norm_eps=params["rms_norm_eps"],
            vocab_size=params["vocab_size"],
            rope_theta=params["rope_theta"],
            moe={
                "num_experts_per_tok": params["num_experts_per_tok"],
                "num_experts": params["num_local_experts"],
            },
        )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, li: int):
        super().__init__()
        self.args = args
        self.li = li
        self.rope_cos: torch.Tensor
        self.rope_sin: torch.Tensor
        self.cache: torch.Tensor
        self.mask: torch.Tensor
        self.prefill_storage_idx: torch.Tensor
        self.decode_storage_idx: torch.Tensor

        self.n_heads: int = args.n_heads
        self.head_dim: int = args.head_dim
        self.sqrt_head_dim = self.head_dim**0.5
        self.n_kv_heads: int = args.n_kv_heads
        self.repeats = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def set_batch_level_args(
        self,
        rope_cos: torch.Tensor,        # [max_len, D/2], float
        rope_sin: torch.Tensor,
        cache: torch.Tensor,
        mask: torch.Tensor,
        prefill_storage_idx: torch.Tensor,
        decode_storage_idx: torch.Tensor,
    ):
        self.rope_cos = rope_cos
        self.rope_sin = rope_sin
        self.cache = cache
        self.mask = mask
        self.prefill_storage_idx = prefill_storage_idx
        self.decode_storage_idx = decode_storage_idx

    def forward(
        self,
        x: torch.Tensor,
        storage_idx: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # fuse rope
        cos_t = self.rope_cos[storage_idx]    # [T, D/2]
        sin_t = self.rope_sin[storage_idx]

        xq = xq.contiguous()   # [B, n_heads,    T, D]
        xk = xk.contiguous()   # [B, n_kv_heads, T, D]

        xq, xk = rope_fused(xq, xk, cos_t, sin_t)
        

        # assumes bsz matches that of cache
        self.cache[0, self.li].index_copy_(dim=-2, index=storage_idx, source=xk)
        self.cache[1, self.li].index_copy_(dim=-2, index=storage_idx, source=xv)
        keys = self.cache[0, self.li]
        values = self.cache[1, self.li]

        # repeat k/v heads if n_kv_heads < n_heads
        keys, values = repeat_kv_fused(keys, values, self.repeats)


        output = F.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=self.mask[storage_idx],
            dropout_p=0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous().reshape(bsz, seqlen, -1)
        return self.wo(output)


class Experts:

    def __init__(self, ws: dict):
        self.ws: dict[str, torch.Tensor] = ws

    def forward(self, li: int, ei: int, x: torch.Tensor) -> torch.Tensor:
        w_gate_up: torch.Tensor = self.ws[f"{li}.{ei}.w_gate_up"].T
        w_down: torch.Tensor = self.ws[f"{li}.{ei}.w_down"].T
        gate_states, up_states = (x @ w_gate_up).chunk(2, dim=-1)
        hidden_states = nn.functional.silu(gate_states) * up_states
        return hidden_states @ w_down


class MoeLayer(nn.Module):
    def __init__(self, args: ModelArgs, li: int, gate: nn.Module, experts: Experts):
        super().__init__()
        self.num_experts: int = args.moe["num_experts"]
        self.num_experts_per_tok: int = args.moe["num_experts_per_tok"]
        self.first_expert = args.moe["first_expert"]
        self.last_expert = args.moe["last_expert"]
        self.glob_li = li + args.first_layer
        self.gate = gate
        self.experts = experts
        self.dummy_zero = torch.zeros(
            (1,), dtype=torch.int64, device=next(iter(experts.ws.values())).device
        )
        self.pinned_offsets = torch.zeros(
            (1 + self.num_experts,), dtype=torch.int64, device="cpu"
        ).pin_memory()

    def prep_ins(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # WARNING: assumes x to be 2D: (batch_size * seq_len, model_dim)
        gate_logits = self.gate(x)  # 仍用 cuBLAS (最快)
        topk_weight, topk_ids = triton_topk_softmax_smallE(
            gate_logits, k=self.num_experts_per_tok, block_e=8  # E<=64 即可
        )
        topk_weight = topk_weight.flatten().unsqueeze(dim=-1)
        cnts = topk_ids.new_zeros((topk_ids.shape[0], self.num_experts))
        cnts.scatter_(1, topk_ids, 1)
        offsets = torch.cat((self.dummy_zero, cnts.sum(dim=0).cumsum(dim=0)))
        idxs = topk_ids.flatten().argsort()
        adj_idxs = torch.div(idxs, self.num_experts_per_tok, rounding_mode="floor")
        return x[adj_idxs], topk_weight[idxs], offsets, adj_idxs

    def experts_infer(
        self,
        sorted_x: torch.Tensor,
        topk_weight: torch.Tensor,
        offsets: torch.Tensor,
        adj_idxs: torch.Tensor,
        next_r: torch.Tensor,
    ) -> torch.Tensor:
        self.pinned_offsets.copy_(offsets)
        expert_offsets = self.pinned_offsets.tolist()

        expert_outs = []
        for ei in range(self.first_expert, self.last_expert + 1):
            l = expert_offsets[ei]
            r = expert_offsets[ei + 1]
            if l == r:
                continue
            expert_outs.append(
                self.experts.forward(
                    self.glob_li,
                    ei,
                    sorted_x[l:r],
                )
            )

        if len(expert_outs):
            l = expert_offsets[self.first_expert]
            r = expert_offsets[self.last_expert + 1]
            expert_outs = torch.cat(expert_outs)
            expert_outs.mul_(topk_weight[l:r])
            next_r.index_add_(0, adj_idxs[l:r], expert_outs)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm_fused(x, self.weight, self.eps)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, li: int, experts: Experts, local_group):
        super().__init__()
        self.li = li  # local layer number if PP is applied
        self.local_group = local_group
        self.attention = Attention(args, li)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.feed_forward = MoeLayer(
            args=args,
            li=li,
            gate=nn.Linear(args.dim, args.moe["num_experts"], bias=False),
            experts=experts,
        )

    # ****************************************************************************************************
    # The section below is necessary since cuda graph only take functions with torch.tensor typed arguments

    # NOTATION for code below
    # h: residual connection
    # r: normal flow
    # SUPPORTED COMBINATIONS:
    # prefill/decode, first/subseq, not-prl/prl attn, not-prl/inter-prl/intra-prl experts

    def prefill_attn(self, x: torch.Tensor):
        return self.attention(
            self.attention_norm(x), self.attention.prefill_storage_idx
        )

    def decode_attn(self, x: torch.Tensor):
        return self.attention(self.attention_norm(x), self.attention.decode_storage_idx)

    def get_routings(self, h: torch.Tensor, r: torch.Tensor, next_h: torch.Tensor):
        # attn residual connection, (batch_size, seq_len, model_dim)
        torch.add(h, r, out=next_h)
        # (batch_size * seq_len, model_dim)
        r = self.ffn_norm(next_h).view(-1, next_h.shape[-1])
        sorted_r, topk_weight, offsets, adj_idxs = self.feed_forward.prep_ins(r)
        return sorted_r, topk_weight, offsets, adj_idxs

    def moe_single_device(self, h: torch.Tensor, r: torch.Tensor):
        return h + r.view(h.shape)  # MoE res-conn

    def moe_inter_allreduce(self, h: torch.Tensor, r: torch.Tensor):
        dist.all_reduce(r, op=dist.ReduceOp.SUM)
        return h + r.view(h.shape)

    def moe_intra_allreduce(self, h: torch.Tensor, r: torch.Tensor):
        dist.all_reduce(r, op=dist.ReduceOp.SUM, group=self.local_group)
        return h + r.view(h.shape)

    # ==================================================
    # PREFILL, SINGLE-DEVICE-ATTN

    def first_prefill_graphable(self, x: torch.Tensor, next_h: torch.Tensor):
        return self.get_routings(x, self.prefill_attn(x), next_h)

    def subseq_prefill_graphable(
        self, h: torch.Tensor, r: torch.Tensor, next_h: torch.Tensor
    ):
        return self.first_prefill_graphable(self.moe_single_device(h, r), next_h)

    def subseq_prefill_graphable_inter_moe(
        self, h: torch.Tensor, r: torch.Tensor, next_h: torch.Tensor
    ):
        return self.first_prefill_graphable(self.moe_inter_allreduce(h, r), next_h)

    def subseq_prefill_graphable_intra_moe(
        self, h: torch.Tensor, r: torch.Tensor, next_h: torch.Tensor
    ):
        return self.first_prefill_graphable(self.moe_intra_allreduce(h, r), next_h)

    # --------------------------------------------------
    # PREFILL, INTRA-TP-ATTN

    def first_prefill_graphable_intra_attn(self, x: torch.Tensor, next_h: torch.Tensor):
        r = self.prefill_attn(x)
        dist.all_reduce(r, op=dist.ReduceOp.SUM, group=self.local_group)
        return self.get_routings(x, r, next_h)

    def subseq_prefill_graphable_intra_attn_inter_moe(
        self, h: torch.Tensor, r: torch.Tensor, next_h: torch.Tensor
    ):
        return self.first_prefill_graphable_intra_attn(
            self.moe_inter_allreduce(h, r), next_h
        )

    def subseq_prefill_graphable_intra_attn_intra_moe(
        self, h: torch.Tensor, r: torch.Tensor, next_h: torch.Tensor
    ):
        return self.first_prefill_graphable_intra_attn(
            self.moe_intra_allreduce(h, r), next_h
        )

    # --------------------------------------------------
    # PREFILL, INTER-TP-ATTN

    def first_prefill_graphable_inter_attn(self, x: torch.Tensor, next_h: torch.Tensor):
        r = self.prefill_attn(x)
        dist.all_reduce(r, op=dist.ReduceOp.SUM)
        return self.get_routings(x, r, next_h)

    def subseq_prefill_graphable_inter_attn_inter_moe(
        self, h: torch.Tensor, r: torch.Tensor, next_h: torch.Tensor
    ):
        return self.first_prefill_graphable_inter_attn(
            self.moe_inter_allreduce(h, r), next_h
        )

    # ==================================================
    # DECODE, SINGLE-DEVICE-ATTN

    def first_decode_graphable(self, x: torch.Tensor, next_h: torch.Tensor):
        return self.get_routings(x, self.decode_attn(x), next_h)

    def subseq_decode_graphable(
        self, h: torch.Tensor, r: torch.Tensor, next_h: torch.Tensor
    ):
        return self.first_decode_graphable(self.moe_single_device(h, r), next_h)

    def subseq_decode_graphable_inter_moe(
        self, h: torch.Tensor, r: torch.Tensor, next_h: torch.Tensor
    ):
        return self.first_decode_graphable(self.moe_inter_allreduce(h, r), next_h)

    def subseq_decode_graphable_intra_moe(
        self, h: torch.Tensor, r: torch.Tensor, next_h: torch.Tensor
    ):
        return self.first_decode_graphable(self.moe_intra_allreduce(h, r), next_h)

    # --------------------------------------------------
    # DECODE, INTRA-TP-ATTN

    def first_decode_graphable_intra_attn(self, x: torch.Tensor, next_h: torch.Tensor):
        r = self.decode_attn(x)
        dist.all_reduce(r, op=dist.ReduceOp.SUM, group=self.local_group)
        return self.get_routings(x, r, next_h)

    def subseq_decode_graphable_intra_attn_inter_moe(
        self, h: torch.Tensor, r: torch.Tensor, next_h: torch.Tensor
    ):
        return self.first_decode_graphable_intra_attn(
            self.moe_inter_allreduce(h, r), next_h
        )

    def subseq_decode_graphable_intra_attn_intra_moe(
        self, h: torch.Tensor, r: torch.Tensor, next_h: torch.Tensor
    ):
        return self.first_decode_graphable_intra_attn(
            self.moe_intra_allreduce(h, r), next_h
        )

    # --------------------------------------------------
    # DECODE, INTER-TP-ATTN

    def first_decode_graphable_inter_attn(self, x: torch.Tensor, next_h: torch.Tensor):
        r = self.decode_attn(x)
        dist.all_reduce(r, op=dist.ReduceOp.SUM)
        return self.get_routings(x, r, next_h)

    def subseq_decode_graphable_inter_attn_inter_moe(
        self, h: torch.Tensor, r: torch.Tensor, next_h: torch.Tensor
    ):
        return self.first_decode_graphable_inter_attn(
            self.moe_inter_allreduce(h, r), next_h
        )

    # ****************************************************************************************************

    def first_forward(
        self,
        x: torch.Tensor,  # (batch_size, seq_len, model_dim)
        graphs: list[torch.cuda.CUDAGraph],
        data: list[tuple[torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # h.shape = (batch_size, seq_len, model_dim)
        h, res_r, topk_weight, offsets, adj_idxs = data[self.li]
        # (h, r, res_r, topk_weight, offsets, adj_idxs)
        next_r = data[self.li + 1][1]
        h.copy_(x)
        graphs[self.li].replay()

        # h.shape = (batch_size * seq_len, model_dim)
        self.feed_forward.experts_infer(res_r, topk_weight, offsets, adj_idxs, next_r)

    def middle_forward(
        self,
        graphs: list[torch.cuda.CUDAGraph],
        data: list[tuple[torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _h, _r, res_r, topk_weight, offsets, adj_idxs = data[self.li]
        # (h, r, res_r, topk_weight, offsets, adj_idxs) or (h, r, out)
        next_r = data[self.li + 1][1]
        graphs[self.li].replay()
        self.feed_forward.experts_infer(res_r, topk_weight, offsets, adj_idxs, next_r)

    def last_forward(
        self,
        graphs: list[torch.cuda.CUDAGraph],
        data: list[tuple[torch.Tensor]],
    ) -> torch.Tensor:
        self.middle_forward(graphs, data)
        graphs[-1].replay()  # last moe-allreduce


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, experts: Experts, comms: list):
        super().__init__()
        self.args: ModelArgs = args
        (
            self.local_group,
            self.local_leader,
            self.prev_stage_leader,
            self.next_stage_leader,
            self.is_first_stage,
            self.is_last_stage,
        ) = comms
        if self.is_first_stage:
            self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        if self.is_last_stage:
            self.norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.layers = nn.ModuleDict(
            {
                str(li): TransformerBlock(
                    args=args,
                    li=li - args.first_layer,
                    experts=experts,
                    local_group=self.local_group,
                )
                for li in range(args.first_layer, args.last_layer + 1)
            }
        )
        self.prefill_in_buffer: torch.Tensor
        self.decode_in_buffer: torch.Tensor
        self.prefill_out_buffer: torch.Tensor
        self.decode_out_buffer: torch.Tensor

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def set_batch_level_args(
        self,
        bsz: int,
        seqlen: int,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        cache: torch.Tensor,
        mask: torch.Tensor,
        prefill_storage_idx: torch.Tensor,
        decode_storage_idx: torch.Tensor,
    ):
        for li in range(self.args.first_layer, self.args.last_layer + 1):
            self.layers[str(li)].attention.set_batch_level_args(
                rope_cos, rope_sin, cache, mask, prefill_storage_idx, decode_storage_idx
            )
        if self.args.has_pp and not self.is_first_stage:
            self.prefill_in_buffer = torch.zeros(
                (bsz, seqlen, self.args.dim),
                dtype=self.dtype,
                device=self.device,
            )
            self.decode_in_buffer = torch.zeros(
                (bsz, 1, self.args.dim),
                dtype=self.dtype,
                device=self.device,
            )
        if self.args.has_pp and not self.is_last_stage:
            self.prefill_out_buffer = torch.zeros(
                (bsz, seqlen, self.args.vocab_size),
                dtype=self.dtype,
                device=self.device,
            )
            self.decode_out_buffer = torch.zeros(
                (bsz, 1, self.args.vocab_size),
                dtype=self.dtype,
                device=self.device,
            )

    def help_draw_graphs(self, bsz: int, seqlen: int, prefill: bool, pool):
        top_k = self.args.moe["num_experts_per_tok"]

        def select_first_graphable(prefill: bool, options: tuple):
            idx = 0
            if not prefill:
                idx += len(options) // 2
            if self.args.inter_parallel_attn:
                idx += 1
            elif self.args.intra_parallel_attn:
                idx += 2
            return options[idx]

        def select_subseq_graphable(prefill: bool, options: tuple):
            idx = 0
            if not prefill:
                idx += len(options) // 2
            if self.args.inter_parallel_attn:
                idx += 3
            elif self.args.intra_parallel_attn:
                idx += 4
            elif self.args.parallel_experts:
                idx += 1
            else:
                return options[idx]
            if self.args.has_pp:
                idx += 1
            return options[idx]

        def select_last_graphable(options: tuple):
            idx = 0
            if self.args.parallel_experts:
                idx += 1
                if self.args.has_pp:
                    idx += 1
            return options[idx]

        def get_ins(for_h: bool = True):
            shape: tuple
            if for_h:
                shape = (bsz, seqlen, self.args.dim)
            else:
                shape = (bsz * seqlen, self.args.dim)
            return torch.ones(
                shape,
                dtype=self.dtype,
                device=self.device,
            )

        def get_outs():
            return (
                torch.ones(
                    (bsz * seqlen * top_k, self.args.dim),
                    dtype=self.dtype,
                    device=self.device,
                ),
                torch.ones(
                    (bsz * seqlen * top_k, 1),
                    dtype=self.dtype,
                    device=self.device,
                ),
                torch.ones(
                    (1 + self.args.moe["num_experts"],),
                    dtype=torch.int64,
                    device=self.device,
                ),
                torch.ones(
                    (bsz * seqlen * top_k,),
                    dtype=torch.int64,
                    device=self.device,
                ),
            )

        graphs = []
        static_data = []
        h = get_ins()
        next_h = get_ins()
        res_r, topk_weight, offsets, adj_idxs = get_outs()
        k = str(self.args.first_layer)
        func = select_first_graphable(
            prefill,
            (
                self.layers[k].first_prefill_graphable,
                self.layers[k].first_prefill_graphable_inter_attn,
                self.layers[k].first_prefill_graphable_intra_attn,
                self.layers[k].first_decode_graphable,
                self.layers[k].first_decode_graphable_inter_attn,
                self.layers[k].first_decode_graphable_intra_attn,
            ),
        )
        # without this causes cublas_status_not_initialized error
        res_r, topk_weight, offsets, adj_idxs = func(h, next_h)
        graphs.append(torch.cuda.CUDAGraph())
        with torch.cuda.graph(graphs[-1], pool=pool):  # share memory pool
            res_r, topk_weight, offsets, adj_idxs = func(h, next_h)
        static_data.append((h, res_r, topk_weight, offsets, adj_idxs))

        for li in range(self.args.first_layer + 1, self.args.last_layer + 1):
            h = next_h
            r = get_ins(False)
            next_h = get_ins()
            res_r, topk_weight, offsets, adj_idxs = get_outs()
            k = str(li)
            func = select_subseq_graphable(
                prefill,
                (
                    self.layers[k].subseq_prefill_graphable,
                    self.layers[k].subseq_prefill_graphable_inter_moe,
                    self.layers[k].subseq_prefill_graphable_intra_moe,
                    self.layers[k].subseq_prefill_graphable_inter_attn_inter_moe,
                    self.layers[k].subseq_prefill_graphable_intra_attn_inter_moe,
                    self.layers[k].subseq_prefill_graphable_intra_attn_intra_moe,
                    self.layers[k].subseq_decode_graphable,
                    self.layers[k].subseq_decode_graphable_inter_moe,
                    self.layers[k].subseq_decode_graphable_intra_moe,
                    self.layers[k].subseq_decode_graphable_inter_attn_inter_moe,
                    self.layers[k].subseq_decode_graphable_intra_attn_inter_moe,
                    self.layers[k].subseq_decode_graphable_intra_attn_intra_moe,
                ),
            )
            graphs.append(torch.cuda.CUDAGraph())
            with torch.cuda.graph(graphs[-1], pool=graphs[-2].pool()):
                res_r, topk_weight, offsets, adj_idxs = func(h, r, next_h)
            static_data.append((h, r, res_r, topk_weight, offsets, adj_idxs))

        h = next_h
        r = get_ins(False)
        out = get_ins()
        func = select_last_graphable(
            (
                self.layers[k].moe_single_device,
                self.layers[k].moe_inter_allreduce,
                self.layers[k].moe_intra_allreduce,
            )
        )
        graphs.append(torch.cuda.CUDAGraph())
        with torch.cuda.graph(graphs[-1], pool=graphs[-2].pool()):
            out = func(h, r)
        static_data.append((h, r, out))

        return graphs, static_data

    def draw_graphs(self, batch_size: int, prefill_len: int):
        with torch.cuda.device(device=self.device):
            prefill_graphs, prefill_data = self.help_draw_graphs(
                batch_size, prefill_len, True, None
            )
            decode_graphs, decode_data = self.help_draw_graphs(
                batch_size, 1, False, prefill_graphs[-1].pool()
            )
        return prefill_graphs, prefill_data, decode_graphs, decode_data

    def reset_graph_data(self, data: list[tuple[torch.Tensor]]):
        for li in range(self.args.last_layer - self.args.first_layer + 1):
            data[li + 1][1].zero_()

    def forward(
        self,
        xs: torch.Tensor,  # .shape = (bsz, seqlen) or (bsz, seqlen, dim)
        graphs: list[torch.cuda.CUDAGraph],
        data: list[tuple[torch.Tensor]],
        prefill: bool,
    ) -> torch.Tensor:
        if self.is_first_stage:
            xs = self.tok_embeddings(xs)
        elif self.args.has_pp:
            # ignore supplied token_ids
            xs = self.prefill_in_buffer if prefill else self.decode_in_buffer
            if WORLD_RANK == self.local_leader:
                for req in dist.batch_isend_irecv(
                    [dist.P2POp(dist.irecv, xs, self.prev_stage_leader)]
                ):
                    req.wait()
            if self.local_group is not None:
                dist.broadcast(xs, self.local_leader, group=self.local_group)

        self.layers[str(self.args.first_layer)].first_forward(xs, graphs, data)
        for li in range(self.args.first_layer + 1, self.args.last_layer):
            self.layers[str(li)].middle_forward(graphs, data)
        self.layers[str(self.args.last_layer)].last_forward(graphs, data)
        ys = data[-1][2]  # (h, r, out)

        if self.is_last_stage:
            ys = self.output(self.norm(ys))
        else:
            if WORLD_RANK == self.local_leader:
                for req in dist.batch_isend_irecv(
                    [dist.P2POp(dist.isend, ys, self.next_stage_leader)]
                ):
                    req.wait()
            ys = self.prefill_out_buffer if prefill else self.decode_out_buffer
        if self.args.has_pp:
            dist.broadcast(ys, WORLD_SIZE - 1)
        return ys.float()


class Mixtral8x7B:

    @staticmethod
    def build(model_path: str, node_id: int, device: torch.device) -> "Mixtral8x7B":
        model_path = Path(model_path)
        non_experts_filename: str
        for filename in [
            "non-experts.pt",
            f"non-experts-{WORLD_RANK}.pt",
            f"non-experts-{node_id}-{LOCAL_RANK}.pt",
        ]:
            if (model_path / filename).is_file():
                non_experts_filename = filename
        experts_filename = f"experts-{WORLD_RANK}.pt"
        if not (model_path / experts_filename).is_file():
            experts_filename = f"experts-{node_id}-{LOCAL_RANK}.pt"

        model_args = ModelArgs.from_hf_config(get_json(model_path / "config.json"))
        non_experts = torch.load(
            model_path / non_experts_filename,
            map_location=device,
            weights_only=True,
            mmap=True,
        )
        experts = torch.load(
            model_path / experts_filename,
            map_location=device,
            weights_only=True,
            mmap=True,
        )

        # expert key structure: "li.ei.wi"
        fli, lli, fei, lei = (
            model_args.n_layers,
            -1,
            model_args.moe["num_experts"],
            -1,
        )
        for k in experts:
            info = k.split(".")
            li, ei = int(info[0]), int(info[1])
            fli = min(li, fli)
            lli = max(li, lli)
            fei = min(ei, fei)
            lei = max(ei, lei)

        model_args.first_layer = fli
        model_args.last_layer = lli
        model_args.moe["first_expert"] = fei
        model_args.moe["last_expert"] = lei

        # check if PP is applied
        is_first_stage = "tok_embeddings.weight" in non_experts
        is_last_stage = "output.weight" in non_experts
        model_args.has_pp = not is_first_stage or not is_last_stage

        # check if EP or TP is applied on experts
        if (
            any(
                f"{model_args.first_layer}.{ei}.w_down" not in experts
                for ei in range(model_args.moe["num_experts"])
            )
            or experts[f"{model_args.first_layer}.0.w_down"].shape[1]
            < model_args.hidden_dim
        ):
            model_args.parallel_experts = True

        # check if TP is applied on attention
        org_wq_out_dim = model_args.n_heads * model_args.head_dim
        loc_wq_out_dim = non_experts[
            f"layers.{model_args.first_layer}.attention.wq.weight"
        ].shape[0]
        attn_tp_size = org_wq_out_dim // loc_wq_out_dim
        model_args.n_heads //= attn_tp_size
        model_args.n_kv_heads //= attn_tp_size
        if attn_tp_size == WORLD_SIZE:
            model_args.inter_parallel_attn = True
        elif attn_tp_size == LOCAL_WORLD_SIZE:
            model_args.intra_parallel_attn = True

        comms: list
        if (
            model_args.has_pp and model_args.parallel_experts
        ) or model_args.intra_parallel_attn:
            global_map = torch.zeros((WORLD_SIZE, 2), dtype=torch.int64, device=device)
            local_map = torch.tensor(
                [node_id, WORLD_RANK], dtype=torch.int64, device=device
            )
            dist.all_gather_into_tensor(global_map, local_map)
            first_node = torch.min(global_map[:, 0]).item()
            last_node = torch.max(global_map[:, 0]).item()
            local_group, local_leader = None, None

            for ni in range(first_node, last_node + 1):
                ranks_on_node = global_map[global_map[:, 0] == ni][:, 1].tolist()
                node_group = dist.new_group(
                    ranks_on_node, backend="nccl", use_local_synchronization=True
                )
                if node_id == ni:
                    local_group = node_group
                    local_leader = min(ranks_on_node)

            prev_node = node_id - 1 if node_id != first_node else last_node
            next_node = node_id + 1 if node_id != last_node else first_node
            prev_stage_lead = torch.min(
                global_map[global_map[:, 0] == prev_node][:, 1]
            ).item()
            next_stage_lead = torch.min(
                global_map[global_map[:, 0] == next_node][:, 1]
            ).item()

            comms = [local_group, local_leader, prev_stage_lead, next_stage_lead]
        else:
            prev_stage_lead = (WORLD_RANK - 1 + WORLD_SIZE) % WORLD_SIZE
            next_stage_lead = (WORLD_RANK + 1) % WORLD_SIZE
            comms = [None, WORLD_RANK, prev_stage_lead, next_stage_lead]
        comms.append(is_first_stage)
        comms.append(is_last_stage)

        with torch.device("meta"):
            model = Transformer(model_args, Experts(experts), comms)
        model.load_state_dict(non_experts, assign=True, strict=True)

        # TODO: refactor
        if model_args.dim == 4096:
            tokenizer = MistralTokenizer.v1()
        elif model_args.dim == 6144:
            tokenizer = MistralTokenizer.v3()

        # ----------------------------
        # Triton kernels warmup (JIT)
        # ----------------------------
        with torch.cuda.device(device):
            dtype = next(model.parameters()).dtype

            # 1) 预热 rope_fused
            T_w = 4
            B_w = 1
            Hq_w = model_args.n_heads
            Hk_w = model_args.n_kv_heads
            D_h = model_args.head_dim

            xq_w = torch.randn(B_w, Hq_w, T_w, D_h, device=device, dtype=dtype)
            xk_w = torch.randn(B_w, Hk_w, T_w, D_h, device=device, dtype=dtype)
            cos_w, sin_w = precompute_rope_cos_sin(
                dim=D_h, end=T_w, theta=model_args.rope_theta, device=device
            )

            _ = rope_fused(xq_w, xk_w, cos_w, sin_w)

            x_w = torch.randn(1, T_w, model_args.dim, device=device, dtype=dtype)

       
            any_block = next(iter(model.layers.values()))
            _ = any_block.attention_norm(x_w)
            _ = any_block.ffn_norm(x_w)

            
            if model.is_last_stage:
                _ = model.norm(x_w)

            torch.cuda.synchronize()
        return Mixtral8x7B(model, tokenizer)

    def __init__(
        self,
        model: Transformer,
        tokenizer: MistralTokenizer,
    ):
        self.model: Transformer = model
        self.tokenizer: MistralTokenizer = tokenizer

    def encode_prompts(self, prompts: list[str]) -> list[list[int]]:
        return [
            self.tokenizer.encode_chat_completion(
                ChatCompletionRequest(messages=[UserMessage(content=p)])
            ).tokens
            for p in prompts
        ]

    def get_cache(
        self, max_batch_size: int, max_seq_len: int, device: torch.device
    ) -> list[torch.Tensor]:
        return torch.empty(
            (
                2,  # key and value
                self.model.args.last_layer - self.model.args.first_layer + 1,
                max_batch_size,
                self.model.args.n_kv_heads,
                max_seq_len,
                self.model.args.head_dim,
            ),
            dtype=torch.bfloat16,
            device=device,
        )

    def clear_cache(self, cache: torch.Tensor):
        cache.zero_()

    def get_mask(self, max_seq_len: int, dtype: torch.dtype, device: torch.device):
        mask = torch.full(
            (max_seq_len, max_seq_len), float("-inf"), dtype=dtype, device=device
        )
        mask = torch.triu(mask, diagonal=1)
        return mask

    def system_sync(self) -> None:
        torch.cuda.synchronize()
        dist.barrier()

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        *,
        max_gen_len: int,
        temperature: float,
        device: torch.device,
        profile: bool = False,
    ) -> tuple[list[str], int, float, int, float]:

        encoded_prompts = self.encode_prompts(prompts)
        min_p_len = min(len(p) for p in encoded_prompts)
        max_p_len = max(len(p) for p in encoded_prompts)
        max_seq_len = max_p_len + max_gen_len
        bsz = len(encoded_prompts)
        pad_id = max(tkn for p in encoded_prompts for tkn in p) + 1
        eos_id = self.tokenizer.instruct_tokenizer.tokenizer.eos_id

        model = self.model.eval()
        cos, sin = precompute_rope_cos_sin(
            dim=self.model.args.head_dim,
            end=8192,
            theta=self.model.args.rope_theta,
            device=device,
        )
        cache = self.get_cache(bsz, max_seq_len, device)
        mask = self.get_mask(max_seq_len, model.dtype, device)
        p_store_idx = torch.arange(min_p_len, dtype=torch.long, device=device)
        d_store_idx = torch.arange(1, dtype=torch.long, device=device)
        model.set_batch_level_args(
            bsz,
            min_p_len,
            cos,      
            sin, 
            cache,
            mask,
            p_store_idx,
            d_store_idx,
        )

        tokens = torch.full((bsz, max_seq_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(encoded_prompts):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=device)
        input_text_mask = tokens != pad_id

        dummy_p_xs = torch.ones((bsz, min_p_len), dtype=torch.long, device=device)
        dummy_d_xs = torch.ones((bsz, 1), dtype=torch.long, device=device)
        n_warmups = 16

        prefill_graphs, prefill_data, decode_graphs, decode_data = model.draw_graphs(
            bsz, min_p_len
        )
        self.system_sync()
        model.reset_graph_data(prefill_data)
        model.reset_graph_data(decode_data)

        # warmup
        for _ in range(n_warmups):
            model.forward(dummy_p_xs, prefill_graphs, prefill_data, True)
            model.reset_graph_data(prefill_data)
        for _ in range(n_warmups):
            model.forward(dummy_d_xs, decode_graphs, decode_data, False)
            model.reset_graph_data(decode_data)
        self.clear_cache(cache)

        self.system_sync()
        tic = time.time()
        prefill_time: float  # in sec
        decode_time: float  # in sec
        if profile:
            torch.cuda.cudart().cudaProfilerStart()

        # notice:
        # 1. it seems that prompts with length < max will generate
        # max_seq_len - len(prompt) tokens
        # 2. when batch size > 1, only the first bsz * min_prompt_len tokens
        # will be processed in parallel. Longer prompts' remaining tokens are
        # evaluated one-by-one with the min prompt's token generation
        for cur_pos in range(min_p_len, max_seq_len):
            if prev_pos == 0:
                graphs, data = prefill_graphs, prefill_data
            else:
                graphs, data = decode_graphs, decode_data
                d_store_idx.copy_(
                    torch.arange(prev_pos, cur_pos, dtype=torch.long, device=device)
                )
            if cur_pos > min_p_len + 1:
                model.reset_graph_data(decode_data)
            logits = model.forward(
                tokens[:, prev_pos:cur_pos],
                graphs,
                data,
                prev_pos == 0,
            )

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, 0.8)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = (
                input_text_mask[:, cur_pos] * tokens[:, cur_pos]
                + ~input_text_mask[:, cur_pos] * next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= ~input_text_mask[:, cur_pos] & (next_token == eos_id)

            # This should cause an implicit host to device sync point that make profiling results accurate
            is_done = all(eos_reached)
            if prev_pos == 0:
                prefill_time = time.time() - tic
                tic = time.time()

            prev_pos = cur_pos
            if is_done:
                break

        # this part is from here:
        # https://github.com/meta-llama/llama3/blob/main/llama/generation.py
        responses = []
        for bi, tkns in enumerate(tokens.tolist()):
            # cut to max_gen_len
            p_len = len(encoded_prompts[bi])
            tkns: list = tkns[p_len : p_len + max_gen_len]
            # cut to after eos tok if any
            try:
                eos_idx = tkns.index(eos_id)
                tkns = tkns[:eos_idx]
            except ValueError:
                pass
            responses.append(self.tokenizer.decode(tkns))

        n_p_tkns = min_p_len * bsz
        n_gen_tkns = (cur_pos - min_p_len) * bsz

        self.system_sync()
        decode_time = time.time() - tic
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.empty_cache()

        return responses, n_p_tkns, n_gen_tkns, prefill_time, decode_time


def main(
    model_path: str,
    node_id: int,
    prompt: str,
    prompt_path: str,
    n_prompts: int = 1,
    batch_size: int = 1,
    max_gen_len: int = 128,
    hide_resp: bool = False,
):
    # assert prompt or (prompt_path and n_prompts and n_prompts > 0)
    # assert n_prompts % batch_size == 0
    prompts: list[str] = None
    if prompt:
        prompts = [prompt]
    else:
        dataset: list[str] = get_json(Path(prompt_path))["prompts"]
        n_repeats = -(n_prompts // -len(dataset))  # ceil division
        prompts = (dataset * n_repeats)[:n_prompts]

    gpu = torch.device(f"cuda:{LOCAL_RANK}")
    dist.init_process_group(
        "nccl", rank=WORLD_RANK, world_size=WORLD_SIZE, device_id=gpu
    )
    model = Mixtral8x7B.build(model_path, node_id, gpu)

    prefill_tps = []
    decode_tps = []
    for start in range(0, n_prompts, batch_size):
        end = start + batch_size
        prompt_batch = prompts[start:end]
        bsz = len(prompt_batch)
        responses, n_p_tkns, n_gen_tkns, prefill_time, decode_time = model.generate(
            prompt_batch,
            max_gen_len=max_gen_len,
            temperature=0.0,
            device=gpu,
            profile=end >= n_prompts,
        )

        if WORLD_RANK == 0:
            prefill_tp = n_p_tkns / prefill_time
            decode_tp = n_gen_tkns / decode_time
            if n_gen_tkns / bsz > max_gen_len * 0.75:
                prefill_tps.append(prefill_tp)
                decode_tps.append(decode_tp)

            print("=" * 20)
            print("PERFORMANCE BREAKDOWN\n")
            print("PROMPT EVALUATION:")
            print(f"token count: {n_p_tkns}")
            print(f"total time in sec(s): {prefill_time:.2f}")
            print(f"throughput: {prefill_tp:.2f} t/s")
            print("TOKEN GENERATION:")
            print(f"token count: {n_gen_tkns}")
            print(f"total time in sec(s): {decode_time:.2f}")
            if n_gen_tkns > 0:
                print(f"throughput: {decode_tp:.2f} t/s")
            else:
                responses = ["" for _ in prompt_batch]
            if not hide_resp:
                print("=" * 20)
                print("INS-N-OUTS")
                print(f"AVG seqlen: {(n_p_tkns / bsz):.2f}")
                for p, resp in zip(prompt_batch, responses):
                    print(f"PROMPT:\n{p}")
                    print(f"RESPONSE:\n{resp}\n")

        time.sleep(3)

    if WORLD_RANK == 0 and len(prefill_tps) > 1:
        print("=" * 20)
        print("RUN STATISTICS")
        print(f"avg prefill throughput: {mean(prefill_tps[1:]):.2f} t/s")
        print(f"avg decode throughput: {mean(decode_tps[1:]):.2f} t/s")
        print(f"bs: {batch_size}")

    dist.barrier()
    # dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--node-id", type=int)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--prompt-path", type=str)
    parser.add_argument("--n-prompts", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--hide-resp", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(DEFAULT_SEED)
    main(
        args.model_path,
        args.node_id or GROUP_RANK,
        args.prompt,
        args.prompt_path,
        args.n_prompts,
        args.batch_size,
        args.max_tokens,
        args.hide_resp,
    )

    # nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --gpu-metrics-devices=all --gpuctxsw=true torchrun --nnodes=1 --node-rank=0 --nproc-per-node=2 --master-addr=10.10.10.1 --master-port=9091 graph_attn_gate.py
