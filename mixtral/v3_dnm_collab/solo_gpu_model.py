#!/home/joe/miniconda3/envs/mixtral/bin/python

# reference: https://github.com/mistralai/mistral-inference
import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from xformers.ops.fmha import memory_efficient_attention  # type: ignore
from xformers.ops.fmha.attn_bias import (  # type: ignore
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
)

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

MEMCPY_COST = 4  # N different expert's calculations on CPU
IN_CACHE_COST = 0.025
ACT_STATS = None


def reset_perf_logs():
    global ACT_STATS
    ACT_STATS = {li: [] for li in range(32)}


def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(
    keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def get_json(file_path: Path) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


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

    @classmethod
    def from_dict(cls, params: dict):
        cls_params = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in params.items() if k in cls_params})


@dataclass
class SimpleInputMetadata:
    # rope absolute positions
    positions: torch.Tensor

    @staticmethod
    def from_seqlens(seqlens: List[int], device: torch.device) -> "SimpleInputMetadata":
        return SimpleInputMetadata(
            positions=torch.cat([torch.arange(0, seqlen) for seqlen in seqlens]).to(
                device=device, dtype=torch.long
            )
        )


@dataclass
class CacheInputMetadata:
    # rope absolute positions
    positions: torch.Tensor
    # where tokens should go in the cache
    cache_positions: torch.Tensor

    # if prefill, use block diagonal causal mask
    # else use causal with padded key mask
    prefill: bool
    mask: AttentionBias
    seqlens: List[int]


def interleave_list(
    l1: List[torch.Tensor], l2: List[torch.Tensor]
) -> List[torch.Tensor]:
    assert len(l1) == len(l2)
    return [v for pair in zip(l1, l2) for v in pair]


class CacheView:
    def __init__(
        self,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        metadata: CacheInputMetadata,
        kv_seqlens: torch.Tensor,
    ):
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.kv_seqlens = kv_seqlens
        self.metadata = metadata

    def update(self, xk: torch.Tensor, xv: torch.Tensor) -> None:
        """
        to_cache_mask masks the last [max_seq_len] tokens in each sequence
        """
        n_kv_heads, head_dim = self.cache_k.shape[-2:]
        flat_cache_k = self.cache_k.view(-1, n_kv_heads, head_dim)
        flat_cache_v = self.cache_v.view(-1, n_kv_heads, head_dim)

        flat_cache_k.index_copy_(0, self.metadata.cache_positions, xk)
        flat_cache_v.index_copy_(0, self.metadata.cache_positions, xv)

    def interleave_kv(
        self, xk: torch.Tensor, xv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is a naive implementation and not optimized for speed.
        """
        assert xk.ndim == xv.ndim == 3  # (B * T, H, D)
        assert xk.shape == xv.shape

        if all([s == 0 for s in self.metadata.seqlens]):
            # No cache to interleave
            return xk, xv

        # Make it a list of [(T, H, D)]
        xk: Tuple[torch.Tensor] = torch.split(xk, self.metadata.seqlens)  # type: ignore
        xv: Tuple[torch.Tensor] = torch.split(xv, self.metadata.seqlens)  # type: ignore
        assert len(xk) == len(
            self.kv_seqlens
        ), f"Batch size is {len(self.kv_seqlens)}, got {len(xk)}"

        # Retrieve cache
        cache_k = [
            cache_k[:seq_len] for cache_k, seq_len in zip(self.cache_k, self.kv_seqlens)
        ]
        cache_v = [
            cache_v[:seq_len] for cache_v, seq_len in zip(self.cache_v, self.kv_seqlens)
        ]

        interleaved_k = interleave_list(cache_k, list(xk))
        interleaved_v = interleave_list(cache_v, list(xv))

        return torch.cat(interleaved_k, dim=0), torch.cat(interleaved_v, dim=0)

    @property
    def max_seq_len(self) -> int:
        return self.cache_k.shape[1]

    @property
    def key(self) -> torch.Tensor:
        return self.cache_k[: len(self.kv_seqlens)]

    @property
    def value(self) -> torch.Tensor:
        return self.cache_v[: len(self.kv_seqlens)]

    @property
    def prefill(self) -> bool:
        return self.metadata.prefill

    @property
    def mask(self) -> AttentionBias:
        return self.metadata.mask


class BufferCache:
    """
    This is an example that implements a buffer cache, allowing for variable length sequences.
    Allocated cache is rectangular which is wasteful (see PagedAttention for better mechanisms)
    """

    def __init__(
        self,
        n_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
    ):
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.cache_k = torch.empty(
            (n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim)
        )
        self.cache_v = torch.empty(
            (n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim)
        )
        # holds the valid length for each batch element in the cache
        self.kv_seqlens: Optional[torch.Tensor] = None

    def get_view(self, layer_id: int, metadata: CacheInputMetadata) -> CacheView:
        assert self.kv_seqlens is not None
        return CacheView(
            self.cache_k[layer_id], self.cache_v[layer_id], metadata, self.kv_seqlens
        )

    def reset(self) -> None:
        self.kv_seqlens = None

    def init_kvseqlens(self, batch_size: int) -> None:
        self.kv_seqlens = torch.zeros(
            (batch_size,), device=self.device, dtype=torch.long
        )

    @property
    def device(self) -> torch.device:
        return self.cache_k.device

    def to(self, device: torch.device, dtype: torch.dtype) -> "BufferCache":
        self.cache_k = self.cache_k.to(device=device, dtype=dtype)
        self.cache_v = self.cache_v.to(device=device, dtype=dtype)

        return self

    def update_seqlens(self, seqlens: List[int]) -> None:
        assert self.kv_seqlens is not None
        self.kv_seqlens += torch.tensor(seqlens, device=self.device, dtype=torch.long)

    def get_input_metadata(self, seqlens: List[int]) -> CacheInputMetadata:
        """
        Get metadata about cache positions
        """
        if self.kv_seqlens is None:
            self.init_kvseqlens(len(seqlens))

        assert isinstance(self.kv_seqlens, torch.Tensor)
        assert len(seqlens) == len(
            self.kv_seqlens
        ), f"Batch size is {len(self.kv_seqlens)}, got {len(seqlens)}, did you forget to reset cache?"
        seqpos = self.kv_seqlens.tolist()

        assert len(seqlens) > 0, seqlens
        cached_elements = torch.tensor(seqlens, device=self.device, dtype=torch.long)

        positions = torch.cat(
            [torch.arange(pos, pos + seqlen) for pos, seqlen in zip(seqpos, seqlens)]
        ).to(device=self.device, dtype=torch.long)
        batch_idx = torch.tensor(
            sum([[i] * seqlen for i, seqlen in enumerate(seqlens)], []),
            device=self.device,
            dtype=torch.long,
        )
        cache_positions = positions + batch_idx * self.max_seq_len

        during_prefill = seqpos[0] == 0
        if during_prefill:
            assert all([pos == 0 for pos in seqpos]), seqpos
            mask = BlockDiagonalCausalMask.from_seqlens(seqlens).make_local_attention(
                self.max_seq_len
            )
        else:
            mask = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
                q_seqlen=seqlens,
                kv_padding=self.max_seq_len,
                kv_seqlen=(self.kv_seqlens + cached_elements)
                .clamp(max=self.max_seq_len)
                .tolist(),
            )

        return CacheInputMetadata(
            positions=positions,
            cache_positions=cache_positions,
            prefill=during_prefill,
            mask=mask,
            seqlens=seqlens,
        )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.head_dim: int = args.head_dim
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[CacheView],
    ) -> torch.Tensor:
        seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if cache is None:
            key, val = xk, xv
        elif cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else:
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(
                seqlen_sum * cache.max_seq_len, self.n_kv_heads, self.head_dim
            )
            val = val.view(
                seqlen_sum * cache.max_seq_len, self.n_kv_heads, self.head_dim
            )

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        output = memory_efficient_attention(
            xq, key, val, None if cache is None else cache.mask
        )
        output = output.view(seqlen_sum, self.n_heads * self.head_dim)

        assert isinstance(output, torch.Tensor)

        return self.wo(output)  # type: ignore


class Experts:
    # tmp design:
    # 1. shared across layers
    # 2. CPU and GPU computations are not overlapped

    def __init__(self, ws: dict):
        self.ws: dict[str, torch.Tensor] = ws

    def forward(self, li: int, ei: int, x: torch.Tensor) -> torch.Tensor:
        w = self.ws[f"{li}.{ei}"].to(x.device, non_blocking=True)
        return (nn.functional.silu(x @ w[0].T) * (x @ w[2].T)) @ w[1]  # type: ignore


class MoeLayer(nn.Module):
    def __init__(self, args: ModelArgs, li: int, gate: nn.Module, experts: Experts):
        super().__init__()
        self.num_experts: int = args.moe["num_experts"]
        self.num_experts_per_tok: int = args.moe["num_experts_per_tok"]
        self.li = li
        self.gate = gate
        self.experts = experts
        self.static_e = 0

    def allocate(
        self, selected_experts: torch.Tensor, inputs: torch.Tensor
    ) -> tuple[list, list, dict]:
        jobs = []
        worthy = []
        unworthy = {}  # initially dict[int, int], later dict[int, torch.Tensor]
        spares = 0
        on_gpu_cnt = 0  # perf_analysis
        for ei in range(self.num_experts):
            batch_idx, nth_expert = torch.where(selected_experts == ei)
            load = (
                1 + (batch_idx.shape[0] - 1) * IN_CACHE_COST
                if batch_idx.shape[0]
                else 0
            )
            jobs.append((batch_idx, nth_expert))
            if ei == self.static_e or load >= MEMCPY_COST:
                worthy.append(ei)
                on_gpu_cnt += batch_idx.shape[0]  # perf_analysis
            elif load > 0:  # filtering out unselected experts
                spares += load
                unworthy[ei] = load

        for ei, load in sorted(
            unworthy.items(), key=lambda item: item[1], reverse=True
        ):
            if spares - load >= MEMCPY_COST:
                worthy.append(ei)
                del unworthy[ei]
                spares -= load + MEMCPY_COST
                on_gpu_cnt += jobs[ei][0].shape[0]  # perf_analysis
            else:
                unworthy[ei] = inputs[jobs[ei][0]].to("cpu")

        ACT_STATS[self.li].append(on_gpu_cnt / (inputs.shape[0] * 2))

        return jobs, worthy, unworthy

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        jobs, worthy, unworthy = self.allocate(selected_experts, inputs)

        # ACT_STATS[self.li].append(weights)  # experiment
        # TODO: static_e should work first to enable more memcpy overlap
        for ei in worthy:
            batch_idx, nth_expert = jobs[ei]
            ey = self.experts.forward(self.li, ei, inputs[batch_idx])
            results[batch_idx] += weights[batch_idx, nth_expert, None] * ey

        cpu_eys: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for ei, ex in unworthy.items():
            batch_idx, nth_expert = jobs[ei]
            cpu_eys.append(
                (
                    batch_idx,
                    weights[batch_idx, nth_expert, None],
                    self.experts.forward(self.li, ei, ex),
                )
            )

        torch.cuda.synchronize()
        for batch_idx, w, ey in cpu_eys:
            results[batch_idx] += w * ey.to(w.device)

        return results


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, li: int, experts: Experts):
        super().__init__()
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.feed_forward = MoeLayer(
            args=args,
            li=li,
            gate=nn.Linear(args.dim, args.moe["num_experts"], bias=False),
            experts=experts,
        )

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, cache: Optional[CacheView]
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, experts: Experts):
        super().__init__()
        self.args = args
        self._precomputed_freqs_cis: torch.Tensor = None
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.layers = nn.ModuleDict(
            {
                str(li): TransformerBlock(args=args, li=li, experts=experts)
                for li in range(args.n_layers)
            }
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def freqs_cis(self) -> torch.Tensor:
        # We cache freqs_cis but need to take care that it is on the right device
        # and has the right dtype (complex64). The fact that the dtype is different
        # from the module's dtype means we cannot register it as a buffer
        if self._precomputed_freqs_cis is None:
            # default to 10**6
            theta = self.args.rope_theta or 1000000.0
            self._precomputed_freqs_cis = precompute_freqs_cis(
                self.args.head_dim, 128_000, theta
            )

        if self._precomputed_freqs_cis.device != self.device:
            self._precomputed_freqs_cis = self._precomputed_freqs_cis.to(
                device=self.device
            )
        return self._precomputed_freqs_cis

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        cache: BufferCache,
    ) -> torch.Tensor:
        (num_toks,) = input_ids.shape
        assert sum(seqlens) == num_toks, (sum(seqlens), num_toks)

        input_metadata = cache.get_input_metadata(seqlens)
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[input_metadata.positions]

        for li in range(self.args.n_layers):
            cache_view = cache.get_view(li, input_metadata)
            h = self.layers[str(li)](h, freqs_cis, cache_view)

        cache.update_seqlens(seqlens)
        outs = self.output(self.norm(h))
        return outs.float()

    @staticmethod
    def load(model_path: Path, gpu: torch.device) -> "Transformer":
        model_args = ModelArgs.from_dict(get_json(model_path / "params.json"))

        non_experts = torch.load(
            model_path / "non-experts.pt",
            map_location=gpu,
            mmap=True,
        )
        experts: dict[str, torch.Tensor] = torch.load(
            model_path / "experts.pt", map_location=torch.device("cpu"), mmap=True
        )

        for li in range(model_args.n_layers):
            # static_e = li % model_args.moe["num_experts"]
            for ei in range(model_args.moe["num_experts"]):
                # experts[f"{li}.{ei}"] = (
                #     experts[f"{li}.{ei}"].to(gpu)
                #     if ei == static_e
                #     else experts[f"{li}.{ei}"].pin_memory()
                # )
                experts[f"{li}.{ei}"] = (
                    experts[f"{li}.{ei}"].to(gpu)
                    if ei == 0
                    else experts[f"{li}.{ei}"].pin_memory()
                )

        with torch.device("meta"):
            model = Transformer(args=model_args, experts=Experts(experts))
        model.load_state_dict(non_experts, assign=True, strict=True)

        return model


@torch.inference_mode()
def generate(
    prompts: List[str],
    tokenizer: MistralTokenizer,
    model: Transformer,
    gpu: torch.device,
    *,
    max_tokens: int,
    max_batch_size: int = 64,
    temperature: float = 0.0,
    eos_id: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[List[str], int, float, int, float]:
    model = model.eval()
    prefill_tic = torch.cuda.Event(enable_timing=True)
    prefill_toc = torch.cuda.Event(enable_timing=True)
    prefill_tic.record()

    encoded_prompts: List[List[int]] = [
        tokenizer.encode_chat_completion(
            ChatCompletionRequest(messages=[UserMessage(content=p)])
        ).tokens
        for p in prompts
    ]
    B, V = len(encoded_prompts), model.args.vocab_size
    seqlens = [len(x) for x in encoded_prompts]

    # Cache
    cache_window = max(seqlens) + max_tokens
    cache = BufferCache(
        model.args.n_layers,
        max_batch_size,
        cache_window,
        model.args.n_kv_heads,
        model.args.head_dim,
    )
    cache.to(device=model.device, dtype=model.dtype)
    cache.reset()

    # prefill / prompt evaluation stage
    reset_perf_logs()  # perf_analysis
    prelogits = model.forward(
        torch.tensor(sum(encoded_prompts, []), device=model.device, dtype=torch.long),
        seqlens=seqlens,
        cache=cache,
    )
    last_positions = torch.tensor(seqlens, device=prelogits.device).cumsum(dim=0) - 1
    last_token_prelogits = prelogits.index_select(0, last_positions)
    prefill_toc.record()
    torch.cuda.synchronize(device=gpu)
    prefill_time = prefill_tic.elapsed_time(prefill_toc) / 1000  # to seconds
    if verbose:  # perf_analysis
        on_gpu_pct = [round(ACT_STATS[li][0], 3) for li in range(32)]
        print("PCT OF EXPERT CALCS ON GPU DURING PREFILL:\n", on_gpu_pct)
        print(f"AVG: {round(mean(on_gpu_pct), 3)}")

    # decode
    reset_perf_logs()  # perf_analysis
    decode_tic = torch.cuda.Event(enable_timing=True)
    decode_toc = torch.cuda.Event(enable_timing=True)
    decode_tic.record()
    generated_tensors = []
    is_finished = torch.tensor([False for _ in range(B)])

    for _ in range(max_tokens):
        next_token = sample(last_token_prelogits, temperature=temperature, top_p=0.8)
        is_finished = is_finished | (next_token == eos_id).cpu()

        if is_finished.all():
            break

        generated_tensors.append(next_token[:, None])
        last_token_prelogits = model.forward(next_token, seqlens=[1] * B, cache=cache)
        assert last_token_prelogits.shape == (B, V)

    generated_tokens: List[List[int]]
    n_gen_tkns = 0
    if generated_tensors:
        generated_tokens = torch.cat(generated_tensors, 1).tolist()
        n_gen_tkns = sum(len(y) - 1 for y in generated_tokens)
    else:
        generated_tokens = []
    responses = [tokenizer.decode(y) for y in generated_tokens]
    decode_toc.record()
    torch.cuda.synchronize(device=gpu)
    decode_time = decode_tic.elapsed_time(decode_toc) / 1000  # to seconds
    if verbose:  # perf_analysis
        on_gpu_pct = [round(mean(ACT_STATS[li]), 3) for li in range(32)]
        print("PCT OF EXPERT CALCS ON GPU DURING DECODE:\n", on_gpu_pct)
        print(f"AVG: {round(mean(on_gpu_pct), 3)}")  # average of averages

    return (
        seqlens,
        responses,
        sum(seqlens),
        prefill_time,
        n_gen_tkns,
        decode_time,
    )


def sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


def main(
    model_path: str,
    prompt: str,
    prompt_path: str,
    n_prompts: int,
    max_tokens: int,
    hide_resp: bool,
):
    assert prompt or (prompt_path and n_prompts and n_prompts > 0)
    gpu_0 = torch.device("cuda:0")
    prompts: list[str] = None
    if prompt:
        prompts = [prompt]
    else:
        dataset: list[str] = get_json(Path(prompt_path))["prompts"]
        n_repeats = -(n_prompts // -len(dataset))  # ceil division
        prompts = (dataset * n_repeats)[:n_prompts]
    tokenizer = MistralTokenizer.v1()
    model = Transformer.load(Path(model_path), gpu_0)

    # warmup
    generate(
        ["hello, how are you?"],
        tokenizer,
        model,
        gpu_0,
        max_tokens=1,
        max_batch_size=len(prompts),
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
    )
    print("finished warming up")

    seqlens, responses, n_p_tkns, prefill_time, n_gen_tkns, decode_time = generate(
        prompts,
        tokenizer,
        model,
        gpu_0,
        max_tokens=max_tokens,
        max_batch_size=len(prompts),
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
        verbose=True,  # perf_analysis
    )
    print("=" * 20)
    print("PERFORMANCE BREAKDOWN\n")
    print("PROMPT EVALUATION:")
    print(f"token count: {n_p_tkns}")
    print(f"total time in sec(s): {prefill_time:.2f}")
    print(f"throughput: {(n_p_tkns / prefill_time):.2f} t/s")
    print("TOKEN GENERATION:")
    print(f"token count: {n_gen_tkns}")
    print(f"total time in sec(s): {decode_time:.2f}")
    if n_gen_tkns > 0:
        print(f"throughput: {(n_gen_tkns / decode_time):.2f} t/s")
    else:
        responses = ["" for _ in prompts]
    if not hide_resp:
        print("=" * 20)
        print("In-n-Outs")
        print(f"AVG seqlen: {mean(seqlens)}")
        print(f"seqlens: {seqlens}\n")
        for p, resp in zip(prompts, responses):
            print(f"PROMPT:\n{p}")
            print(f"RESPONSE:\n{resp}\n")

    # top1_avgs, top2_avgs = [], []
    # for li in range(32):
    #     score_avgs = torch.mean(torch.cat(ACT_STATS[li], axis=0).T, dim=1).tolist()
    #     assert len(score_avgs) == 2
    #     top1_avgs.append(score_avgs[0])
    #     top2_avgs.append(score_avgs[1])

    # print()
    # print(f"top-1 layer-wise score average: (model-wise: {mean(top1_avgs)})")
    # print(top1_avgs)
    # print(f"top-2 layer-wise score average: (model-wise: {mean(top2_avgs)})")
    # print(top2_avgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--prompt-path", type=str)
    parser.add_argument("--n-prompts", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--hide-resp", action="store_true")
    args = parser.parse_args()

    main(
        args.model_path,
        args.prompt,
        args.prompt_path,
        args.n_prompts,
        args.max_tokens,
        args.hide_resp,
    )