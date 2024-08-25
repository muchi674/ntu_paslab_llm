import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import safetensors.torch

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


@dataclass
class ModelArgs:
    # follows hf weights config.json
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_local_experts: int
    num_experts_per_tok: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int

    # calculated after json.load()
    head_dim: int = 0

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

        self.n_heads: int = args.num_attention_heads
        self.head_dim: int = args.head_dim
        self.n_kv_heads: int = args.num_key_value_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.head_dim**-0.5

        self.wq = nn.Linear(args.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            args.hidden_size, self.n_kv_heads * self.head_dim, bias=False
        )
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=False)

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
    # tmp design: shared across layers

    def __init__(self, ws: dict):
        self.ws = ws

    def forward(self, li: int, ei: int, x: torch.Tensor) -> torch.Tensor:
        w = self.ws[f"{li}.{ei}"]
        return (nn.functional.silu(x @ w[0].T) * (x @ w[2].T)) @ w[1]  # type: ignore


class MoeLayer(nn.Module):
    def __init__(self, args: ModelArgs, li: int, gate: nn.Module, experts: Experts):
        super().__init__()
        self.num_local_experts: int = args.num_local_experts
        self.num_experts_per_tok: int = args.num_experts_per_tok
        self.li = li
        self.gate = gate
        self.experts = experts

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)

        for ei in range(self.num_local_experts):
            batch_idx, nth_expert = torch.where(selected_experts == ei)
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
        self.attention_norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.feed_forward = MoeLayer(
            args=args,
            li=li,
            gate=nn.Linear(args.hidden_size, args.num_local_experts, bias=False),
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
        self.vocab_size = args.vocab_size
        self._precomputed_freqs_cis: torch.Tensor = None
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.output = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.layers = nn.ModuleDict(
            {
                str(li): TransformerBlock(args=args, li=li, experts=experts)
                for li in range(args.num_hidden_layers)
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
                self.args.hidden_size // self.args.num_attention_heads, 128_000, theta
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

        for li, layer in self.layers.items():
            cache_view = cache.get_view(li, input_metadata)
            h = layer(h, freqs_cis, cache_view)

        cache.update_seqlens(seqlens)
        outs = self.output(self.norm(h))
        return outs.float()

    @staticmethod
    def load(model_path: Path) -> "Transformer":
        with open(model_path / "config.json", "r") as f:
            model_args = ModelArgs.from_dict(json.load(f))
        model_args.head_dim = model_args.hidden_size // model_args.num_attention_heads

        gpu_0 = torch.device("cuda:0")
        non_experts = safetensors.torch.load_file(
            model_path / "non-experts.safetensors", device=gpu_0
        )
        experts = safetensors.torch.load_file(
            model_path / "non-experts.safetensors", device="cpu"
        )

        with torch.device("meta"):
            model = Transformer(args=model_args, experts=experts)
        model.load_state_dict(non_experts, assign=True, strict=True)

        return model


@torch.inference_mode()
def generate(
    encoded_prompts: List[List[int]],
    model: Transformer,
    *,
    max_tokens: int,
    max_batch_size: int = 64,
    temperature: float = 0.0,
    eos_id: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[float]]]:
    model = model.eval()
    B, V = len(encoded_prompts), model.args.vocab_size

    seqlens = [len(x) for x in encoded_prompts]

    # Cache
    cache_window = max(seqlens) + max_tokens
    cache = BufferCache(
        model.args.num_hidden_layers,
        max_batch_size,
        cache_window,
        model.args.num_key_value_heads,
        model.args.head_dim,
    )
    cache.to(device=model.device, dtype=model.dtype)
    cache.reset()

    # prefill
    prelogits = model.forward(
        torch.tensor(sum(encoded_prompts, []), device=model.device, dtype=torch.long),
        seqlens=seqlens,
        cache=cache,
    )
    last_positions = torch.tensor(seqlens, device=prelogits.device).cumsum(dim=0) - 1
    last_token_prelogits = prelogits.index_select(0, last_positions)

    # decode
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
    if generated_tensors:
        generated_tokens = torch.cat(generated_tensors, 1).tolist()
    else:
        generated_tokens = []

    return generated_tokens


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


def main(model_path: str, prompt: str, max_tokens: int):
    tokenizer = MistralTokenizer.v1()
    completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
    tokens = tokenizer.encode_chat_completion(completion_request).tokens

    model = Transformer.load(Path(model_path))
    out_tkns = generate(
        tokens,
        model,
        max_tokens=max_tokens,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
    )
    result = tokenizer.decode(out_tkns[0])
    print(f"RESPONSE:\n{result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--max-tokens", type=int)
    args = parser.parse_args()

    main(args.model_path, args.prompt, args.max_tokens)
