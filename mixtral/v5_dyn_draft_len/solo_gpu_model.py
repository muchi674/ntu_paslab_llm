#!/home/joe/miniconda3/envs/mixtral/bin/python

# reference: https://github.com/mistralai/mistral-inference
import argparse
import inspect
import json
import time
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
    BlockDiagonalMask,
)

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


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

    def update_seqlens(self, seqlens: List[int], deducts: bool = False) -> None:
        assert self.kv_seqlens is not None
        adjustments = torch.tensor(seqlens, device=self.device, dtype=torch.long)
        if deducts:
            adjustments *= -1
        self.kv_seqlens += adjustments

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

        first_prefill = seqpos[0] == 0
        subsequent_prefill = any(seqlen > 1 for seqlen in seqlens)
        if first_prefill:
            assert all([pos == 0 for pos in seqpos]), seqpos
            mask = BlockDiagonalCausalMask.from_seqlens(seqlens).make_local_attention(
                self.max_seq_len
            )
        elif subsequent_prefill:
            mask = BlockDiagonalMask.from_seqlens(
                q_seqlen=seqlens,
                kv_seqlen=[
                    s + cached_s.clamp(max=self.max_seq_len).item()
                    for (s, cached_s) in zip(seqlens, self.kv_seqlens)
                ],
            ).make_local_attention_from_bottomright(self.max_seq_len)
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
            prefill=first_prefill or subsequent_prefill,
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

    def get_attention_score(
        self,
        key: torch.tensor,
        val: torch.tensor,
        seqlen_sum: int,
        cache: Optional[CacheView],
    ):
        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        output = memory_efficient_attention(
            xq, key, val, None if cache is None else cache.mask
        )
        output = output.view(seqlen_sum, self.n_heads * self.head_dim)

        assert isinstance(output, torch.Tensor)
        return output

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[CacheView],
        draft_x: torch.Tensor = None,
    ) -> torch.Tensor:
        if draft_x:
            x = torch.cat((x, draft_x), dim=0)
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
            # draft_x should only appear here
            cache.update(xk[1:], xv[1:])
            key, val = cache.key, cache.value
            key = key.view(
                seqlen_sum * cache.max_seq_len, self.n_kv_heads, self.head_dim
            )
            val = val.view(
                seqlen_sum * cache.max_seq_len, self.n_kv_heads, self.head_dim
            )

        output = self.get_attention_score(key, val, seqlen_sum, cache)
        return self.wo(output)  # type: ignore


class Experts:
    # tmp design:
    # 1. shared across layers
    # 2. CPU and GPU computations are not overlapped

    def __init__(self, ws: dict):
        self.ws = ws

    def forward(self, li: int, ei: int, x: torch.Tensor) -> torch.Tensor:
        w: torch.Tensor = self.ws[f"{li}.{ei}"]
        ex = x.to(w.device)
        ey = (nn.functional.silu(ex @ w[0].T) * (ex @ w[2].T)) @ w[1]
        return ey.to(x.device)  # type: ignore


class MoeLayer(nn.Module):
    def __init__(self, args: ModelArgs, li: int, gate: nn.Module, experts: Experts):
        super().__init__()
        self.num_experts: int = args.moe["num_experts"]
        self.num_experts_per_tok: int = args.moe["num_experts_per_tok"]
        self.li = li
        self.gate = gate
        self.experts = experts

    def forward(self, inputs: torch.Tensor, drafting: bool) -> torch.Tensor:
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(
            gate_logits, self.num_experts_per_tok if not drafting else 1
        )

        # weights, selected_experts = torch.topk(
        #     gate_logits, self.num_experts_per_tok if not drafting else self.num_experts
        # )
        # # hardcoded logic
        # if drafting:
        #     bi = torch.where(selected_experts[:, 0, None] != 0)[0]
        #     selected_experts[bi, 1] = 0
        #     weights[bi, 1] = gate_logits[bi, 0]
        #     weights = weights[:,:2]
        #     selected_experts = selected_experts[:,:2]

        # weights, selected_experts = torch.topk(
        #     gate_logits, self.num_experts_per_tok if not drafting or self.li < 7 else 1
        # )
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)

        results = torch.zeros_like(inputs)
        for ei in range(self.num_experts):
            batch_idx, nth_expert = torch.where(selected_experts == ei)
            if torch.numel(batch_idx) == 0:
                continue
            ey = self.experts.forward(self.li, ei, inputs[batch_idx])
            results[batch_idx] += weights[batch_idx, nth_expert, None] * ey
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
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: Optional[CacheView],
        drafting: bool,
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h), drafting)
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
        drafting: bool = False,
    ) -> torch.Tensor:
        (num_toks,) = input_ids.shape
        assert sum(seqlens) == num_toks, (sum(seqlens), num_toks)

        input_metadata = cache.get_input_metadata(seqlens)
        h = self.tok_embeddings(input_ids)
        freqs_cis = self.freqs_cis[input_metadata.positions]

        for li in range(self.args.n_layers):
            cache_view = cache.get_view(li, input_metadata)
            h = self.layers[str(li)](h, freqs_cis, cache_view, drafting)

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
            weights_only=True,
        )
        experts = torch.load(
            model_path / "experts.pt",
            map_location=torch.device("cpu"),
            mmap=True,
            weights_only=True,
        )

        for li in range(model_args.n_layers):
            experts[f"{li}.0"] = experts[f"{li}.0"].to(gpu)

        # for li in range(7):
        #     for ei in range(8):
        #         experts[f"{li}.{ei}"] = experts[f"{li}.{ei}"].to(gpu)

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
    draft_seq_len: int = 8,
    temperature: float = 0.6,
    top_k: int = -1,
    top_p: float = 0.9,
    eos_id: Optional[int] = None,
) -> tuple[list[int], list[str], int, float, int, float, float, float, float, float]:
    model = model.eval()
    prefill_tic = torch.cuda.Event(enable_timing=True)
    prefill_toc = torch.cuda.Event(enable_timing=True)
    prefill_tic.record()

    greedy = temperature == 0
    encoded_prompts: List[List[int]] = [
        tokenizer.encode_chat_completion(
            ChatCompletionRequest(messages=[UserMessage(content=p)])
        ).tokens
        for p in prompts
    ]
    B, V = len(encoded_prompts), model.args.vocab_size
    seqlens = [len(x) for x in encoded_prompts]

    # Cache
    cache_window = max(seqlens) + max_tokens + draft_seq_len
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
    logits = model.forward(
        torch.tensor(sum(encoded_prompts, []), device=model.device, dtype=torch.long),
        seqlens=seqlens,
        cache=cache,
    )
    last_positions = torch.tensor(seqlens, device=logits.device).cumsum(dim=0) - 1
    logits = logits.index_select(0, last_positions)
    logits = norm_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    verify_context = sample(logits, greedy=greedy)

    prefill_toc.record()
    torch.cuda.synchronize(device=gpu)
    prefill_time = prefill_tic.elapsed_time(prefill_toc) / 1000  # to seconds

    # decode
    decode_tic = torch.cuda.Event(enable_timing=True)
    decode_toc = torch.cuda.Event(enable_timing=True)
    sub_tic = torch.cuda.Event(enable_timing=True)
    sub_toc = torch.cuda.Event(enable_timing=True)
    draft_lats = []
    target_lats = []
    verify_lats = []
    decode_tic.record()

    curr_gen_lens = [1] * B
    acceptance_lens = []
    accepted_tokens = [[verify_context[None, i]] for i in range(B)]
    is_finished = torch.tensor([False for _ in range(B)], device=model.device)
    is_finished = is_finished | (verify_context == eos_id)

    while max(curr_gen_lens) < max_tokens and not is_finished.all():
        sub_tic.record()

        # context.shape = (batch_size, 1)
        draft_probs = []
        draft_tokens = []
        draft_context = verify_context
        for _ in range(draft_seq_len):
            logits = model.forward(
                draft_context, seqlens=[1] * B, cache=cache, drafting=True
            )
            logits = norm_logits(
                logits, temperature=temperature, top_k=top_k, top_p=top_p
            )  # .shape = (batch_size, vocab_dim)
            draft_context = sample(logits, greedy=greedy)
            draft_probs.append(logits)
            draft_tokens.append(draft_context[:, None])

        sub_toc.record()
        torch.cuda.synchronize(device=gpu)
        draft_lats.append(sub_tic.elapsed_time(sub_toc))
        sub_tic.record()

        draft_probs = torch.stack(draft_probs, dim=1)
        draft_tokens = torch.cat(draft_tokens, dim=1)
        verify_context = torch.flatten(
            torch.cat((verify_context[:, None], draft_tokens), dim=1)
        )
        cache.update_seqlens([draft_seq_len] * B, deducts=True)
        verify_logits = model.forward(
            verify_context, seqlens=[1 + draft_seq_len] * B, cache=cache
        )

        sub_toc.record()
        torch.cuda.synchronize(device=gpu)
        target_lats.append(sub_tic.elapsed_time(sub_toc))
        sub_tic.record()

        if greedy:
            curr_accept_lens, verify_context = verify_greedy(
                draft_tokens,
                verify_logits,
                is_finished,
                accepted_tokens,
                B,
                draft_seq_len,
                eos_id,
            )
        else:
            curr_accept_lens, verify_context = verify_non_greedy(
                draft_tokens,
                draft_probs,
                verify_logits,
                is_finished,
                accepted_tokens,
                B,
                draft_seq_len,
                temperature,
                top_k,
                top_p,
                eos_id,
            )

        sub_toc.record()
        torch.cuda.synchronize(device=gpu)
        verify_lats.append(sub_tic.elapsed_time(sub_toc))

        acceptance_lens.append(curr_accept_lens)
        cache.update_seqlens(
            [1 + draft_seq_len - val for val in curr_accept_lens], deducts=True
        )
        for bi, val in enumerate(curr_accept_lens):
            curr_gen_lens[bi] += val

    responses = []
    n_gen_tkns = 0
    for tkns in accepted_tokens:
        if len(tkns) > 0:
            tkns = torch.cat(tkns, dim=0).tolist()
            n_gen_tkns += len(tkns) - 1
        responses.append(tokenizer.decode(tkns))

    decode_toc.record()
    torch.cuda.synchronize(device=gpu)
    decode_time = decode_tic.elapsed_time(decode_toc) / 1000  # to seconds

    return (
        seqlens,
        responses,
        sum(seqlens),
        prefill_time,
        n_gen_tkns,
        decode_time,
        mean(draft_lats) if len(draft_lats) > 0 else None,  # in ms
        mean(target_lats) if len(target_lats) > 0 else None,  # in ms
        mean(verify_lats) if len(verify_lats) > 0 else None,  # in ms
        (
            torch.tensor(acceptance_lens, device="cpu")
            .to(torch.float32)
            .flatten()
            .mean(dim=0)
            .item()
            if len(acceptance_lens) > 0
            else None
        ),
    )


def verify_greedy(
    draft_tokens: torch.Tensor,
    verify_logits: torch.Tensor,
    is_finished: torch.Tensor,
    accepted_tokens: list,
    batch_size: int,
    draft_seq_len: int,
    eos_id: int,
) -> tuple[list, torch.Tensor]:
    # draft_tokens.shape = (batch_size, draft_seq_len)
    # verify_logits.shape = (batch_size * (draft_seq_len + 1), vocab_dim)
    # Compare draft tokens against argmax(logits) for each position in the sequence
    verify_logits = verify_logits.unflatten(
        dim=0, sizes=(batch_size, draft_seq_len + 1)
    )
    verify_tokens = sample(verify_logits[:, :-1], greedy=True)
    acceptance = (draft_tokens == verify_tokens).to(torch.int32)
    acceptance_lens = torch.cumprod(acceptance, dim=1).sum(dim=1).tolist()
    next_verify_context = []

    for bi, accept_len in enumerate(acceptance_lens.copy()):
        res = draft_tokens[bi, :accept_len]
        if accept_len == 0:
            res = verify_tokens[bi, 0][None]
            acceptance_lens[bi] += 1
        elif accept_len == draft_seq_len:
            res = torch.cat((res, sample(verify_logits[None, bi, -1], greedy=True)))
            acceptance_lens[bi] += 1

        accepted_tokens[bi].append(res)
        next_verify_context.append(res[None, -1])
        is_finished[bi] = is_finished[bi] | torch.any(res == eos_id)

    return acceptance_lens, torch.cat(next_verify_context, dim=0)


def verify_non_greedy(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    verify_logits: torch.Tensor,
    is_finished: torch.Tensor,
    accepted_tokens: list,
    batch_size: int,
    draft_seq_len: int,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_id: int,
) -> tuple[list, torch.Tensor]:
    # draft_tokens.shape = (batch_size, draft_seq_len)
    # draft_probs.shape = (batch_size, draft_seq_len, vocab_dim)
    # verify_logits.shape = (batch_size * (draft_seq_len + 1), vocab_dim)
    device = draft_tokens.device
    verify_probs = norm_logits(
        verify_logits, temperature=temperature, top_k=top_k, top_p=top_p
    ).unflatten(dim=0, sizes=(batch_size, draft_seq_len + 1))
    acceptance_lens = []
    next_verify_context = []

    # for each in batch
    for bi, dts, dps, vps in zip(
        range(batch_size), draft_tokens, draft_probs, verify_probs
    ):
        # for each position in draft sequence
        res = []
        count = 0
        for i, draft_prob, verify_prob in zip(dts, dps, vps):
            r = torch.rand(1, device=device)
            if r < torch.min(
                torch.tensor([1], device=device), (verify_prob[i] / draft_prob[i])
            ):
                count += 1
                res.append(i[None])
            else:
                resampled = sample(max_fn(verify_prob - draft_prob))
                res.append(resampled)
                break

        if count == draft_seq_len:
            i = sample(vps[-1])
            res.append(i)

        acceptance_lens.append(len(res))
        next_verify_context.append(res[-1])

        res = torch.cat(res, dim=0)
        accepted_tokens[bi].append(res)
        is_finished[bi] = is_finished[bi] | torch.any(res == eos_id)

    return acceptance_lens, torch.cat(next_verify_context, dim=0)


# modified from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    # unlike Meta Llama's reference implementation,
    # this version preserves the order of logits throughput computation
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float("-inf")
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float("-inf")
    return logits


# modified from https://github.com/Infini-AI-Lab/TriForce/blob/main/utils/sampling.py
# -----TriForce start-----


def norm_logits(
    logits: torch.Tensor, temperature: float = 0.6, top_k: int = -1, top_p: float = 0.9
) -> torch.Tensor:
    if temperature == 0:
        return logits

    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)

    probs = F.softmax(logits, dim=-1)
    return probs


def sample(probs: torch.Tensor, num_samples: int = 1, greedy: bool = False):
    # probs.shape = (batch_size, vocab_dim)
    if greedy:
        # input is actually logits (not yet normalized to probability distribution)
        return torch.argmax(probs, dim=-1)  # lowers one dimension
    return torch.multinomial(probs, num_samples=num_samples, replacement=True).reshape(
        -1
    )


def max_fn(x: torch.Tensor):
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True)
    if x_max_sum == 0:
        print(x.max(), x.min(), x.shape)
    return x_max / x_max_sum


# -----TriForce end-----


def main(
    model_path: str,
    prompt: str,
    prompt_path: str,
    n_prompts: int,
    batch_size: int,
    max_tokens: int,
    hide_resp: bool,
):
    assert prompt or (prompt_path and n_prompts and n_prompts > 0)
    assert n_prompts % batch_size == 0
    gpu = torch.device("cuda:0")
    prompts: list[str] = None
    if prompt:
        prompts = [prompt]
    else:
        dataset: list[str] = get_json(Path(prompt_path))["prompts"]
        n_repeats = -(n_prompts // -len(dataset))  # ceil division
        prompts = (dataset * n_repeats)[:n_prompts]
    tokenizer = MistralTokenizer.v1()
    model = Transformer.load(Path(model_path), gpu)

    prefill_tps = []
    decode_tps = []
    avg_draft_lats = []
    avg_target_lats = []
    avg_verify_lats = []
    avg_accept_lens = []

    start = 0
    for end in range(batch_size, n_prompts + 1, batch_size):
        prompt_batch = prompts[start:end]

        # warmup
        generate(
            ["hello, how are you?"],
            tokenizer,
            model,
            gpu,
            max_tokens=1,
            max_batch_size=len(prompt_batch),
            temperature=0,
            draft_seq_len=6,
            eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
        )

        (
            seqlens,
            responses,
            n_p_tkns,
            prefill_time,
            n_gen_tkns,
            decode_time,
            avg_draft_lat,
            avg_target_lat,
            avg_verify_lat,
            avg_accept_len,
        ) = generate(
            prompt_batch,
            tokenizer,
            model,
            gpu,
            max_tokens=max_tokens,
            max_batch_size=len(prompt_batch),
            temperature=0,
            draft_seq_len=6,
            eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
        )

        prefill_tp = n_p_tkns / prefill_time
        decode_tp = n_gen_tkns / decode_time
        prefill_tps.append(prefill_tp)
        decode_tps.append(decode_tp)
        if avg_draft_lat is not None:
            avg_draft_lats.append(avg_draft_lat)
            avg_target_lats.append(avg_target_lat)
            avg_verify_lats.append(avg_verify_lat)
            avg_accept_lens.append(avg_accept_len)

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
        print("SPECULATIVE DECODING")
        if avg_draft_lat is not None:
            print(f"avg draft latency: {avg_draft_lat:.2f} ms")
            print(f"avg target latency: {avg_target_lat:.2f} ms")
            print(f"avg verify latency: {avg_verify_lat:.2f} ms")
            print(f"avg acceptance length: {avg_accept_len:.2f}")
        else:
            print("skipped")
        if not hide_resp:
            print("=" * 20)
            print("INS-N-OUTS")
            print(f"AVG seqlen: {mean(seqlens)}")
            print(f"seqlens: {seqlens}\n")
            for p, resp in zip(prompt_batch, responses):
                print(f"PROMPT:\n{p}")
                print(f"RESPONSE:\n{resp}\n")

        start = end
        time.sleep(45)

    print("=" * 20)
    print("RUN STATISTICS")
    print(f"avg prefill throughput: {mean(prefill_tps):.2f} t/s")
    print(f"avg decode throughput: {mean(decode_tps):.2f} t/s")
    if len(avg_draft_lats) > 0:
        print(f"avg draft latency: {mean(avg_draft_lats):.2f} ms")
        print(f"avg target latency: {mean(avg_target_lats):.2f} ms")
        print(f"avg verify latency: {mean(avg_verify_lats):.2f} ms")
        print(f"avg acceptance length: {mean(avg_accept_lens):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--prompt-path", type=str)
    parser.add_argument("--n-prompts", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--hide-resp", action="store_true")
    args = parser.parse_args()

    main(
        args.model_path,
        args.prompt,
        args.prompt_path,
        args.n_prompts,
        args.batch_size,
        args.max_tokens,
        args.hide_resp,
    )
