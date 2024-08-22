import json
import logging
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple, Type, Union

import safetensors.torch

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

        self.n_heads: int = args.num_attention_heads
        self.head_dim: int = args.hidden_size // args.num_attention_heads
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


# class FeedForward(nn.Module):
#     def __init__(self, args: TransformerArgs):
#         super().__init__()

#         MaybeLora = maybe_lora(args)
#         self.w1 = MaybeLora(args.dim, args.hidden_dim, bias=False)
#         self.w2 = MaybeLora(args.hidden_dim, args.dim, bias=False)
#         self.w3 = MaybeLora(args.dim, args.hidden_dim, bias=False)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))  # type: ignore

# class MoeLayer(nn.Module):
#     def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
#         super().__init__()
#         assert len(experts) > 0
#         self.experts = nn.ModuleList(experts)
#         self.gate = gate
#         self.args = moe_args

#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#         gate_logits = self.gate(inputs)
#         weights, selected_experts = torch.topk(
#             gate_logits, self.args.num_experts_per_tok
#         )
#         weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
#         results = torch.zeros_like(inputs)
#         for i, expert in enumerate(self.experts):
#             batch_idx, nth_expert = torch.where(selected_experts == i)
#             results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
#                 inputs[batch_idx]
#             )
#         return results


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
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.ffn_norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args

        self.feed_forward: nn.Module
        self.feed_forward = MoeLayer(
            experts=[FeedForward(args=args) for _ in range(args.moe.num_experts)],
            gate=nn.Linear(args.hidden_size, args.moe.num_experts, bias=False),
            moe_args=args.moe,
        )

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, cache: Optional[CacheView]
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cis, cache)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


# class Transformer(ModelBase, LoRALoaderMixin):
#     def __init__(
#         self,
#         args: TransformerArgs,
#         pipeline_rank: int = 0,
#         num_pipeline_ranks: int = 1,
#     ):
#         super().__init__()
#         self.args = args
#         self.vocab_size = args.vocab_size
#         self.n_layers = args.n_layers
#         self._precomputed_freqs_cis: Optional[torch.Tensor] = None
#         assert self.vocab_size > 0
#         assert pipeline_rank < num_pipeline_ranks, (pipeline_rank, num_pipeline_ranks)
#         self.pipeline_rank = pipeline_rank
#         self.num_pipeline_ranks = num_pipeline_ranks
#         # Modules specific to some ranks:
#         self.tok_embeddings: Optional[nn.Embedding] = None
#         self.norm: Optional[RMSNorm] = None
#         self.output: Optional[nn.Linear] = None
#         if pipeline_rank == 0:
#             self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
#         if pipeline_rank == num_pipeline_ranks - 1:
#             self.norm = RMSNorm(args.dim, eps=args.norm_eps)
#             self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
#         # Initialize all layers but slice off those not of this rank.
#         layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
#         num_layers_per_rank = math.ceil(self.n_layers / self.num_pipeline_ranks)
#         offset = self.pipeline_rank * num_layers_per_rank
#         end = min(self.n_layers, offset + num_layers_per_rank)
#         self.layers = nn.ModuleDict({str(i): layers[i] for i in range(offset, end)})
#         self.n_local_layers = len(self.layers)

#     @property
#     def dtype(self) -> torch.dtype:
#         return next(self.parameters()).dtype

#     @property
#     def device(self) -> torch.device:
#         return next(self.parameters()).device

#     @property
#     def freqs_cis(self) -> torch.Tensor:
#         # We cache freqs_cis but need to take care that it is on the right device
#         # and has the right dtype (complex64). The fact that the dtype is different
#         # from the module's  dtype means we cannot register it as a buffer
#         if self._precomputed_freqs_cis is None:
#             # default to 10**6
#             theta = self.args.rope_theta or 1000000.0
#             self._precomputed_freqs_cis = precompute_freqs_cis(
#                 self.args.head_dim, 128_000, theta
#             )

#         if self._precomputed_freqs_cis.device != self.device:
#             self._precomputed_freqs_cis = self._precomputed_freqs_cis.to(
#                 device=self.device
#             )
#         return self._precomputed_freqs_cis

#     def forward_partial(
#         self,
#         input_ids: torch.Tensor,
#         seqlens: List[int],
#         cache: Optional[BufferCache] = None,
#     ) -> torch.Tensor:
#         """Local forward pass.

#         If doing pipeline parallelism, this will return the activations of the last layer of this stage.
#         For the last stage, this will return the normalized final embeddings.
#         """
#         assert (
#             len(seqlens) <= self.args.max_batch_size
#         ), f"Max batch size is {self.args.max_batch_size}, got batch size of {len(seqlens)}"
#         (num_toks,) = input_ids.shape
#         assert sum(seqlens) == num_toks, (sum(seqlens), num_toks)

#         input_metadata: Union[CacheInputMetadata, SimpleInputMetadata]

#         if cache is not None:
#             input_metadata = cache.get_input_metadata(seqlens)
#         else:
#             input_metadata = SimpleInputMetadata.from_seqlens(seqlens, self.device)

#         if self.pipeline_rank == 0:
#             assert self.tok_embeddings is not None
#             h = self.tok_embeddings(input_ids)
#         else:
#             h = torch.empty(
#                 num_toks, self.args.dim, device=self.device, dtype=self.dtype
#             )
#             torch.distributed.recv(h, src=self.pipeline_rank - 1)

#         freqs_cis = self.freqs_cis[input_metadata.positions]

#         for local_layer_id, layer in enumerate(self.layers.values()):
#             if cache is not None:
#                 assert input_metadata is not None
#                 assert isinstance(input_metadata, CacheInputMetadata)
#                 cache_view = cache.get_view(local_layer_id, input_metadata)
#             else:
#                 cache_view = None
#             h = layer(h, freqs_cis, cache_view)

#         if cache is not None:
#             cache.update_seqlens(seqlens)
#         if self.pipeline_rank < self.num_pipeline_ranks - 1:
#             torch.distributed.send(h, dst=self.pipeline_rank + 1)
#             return h  # type: ignore
#         else:
#             # Last rank has a final normalization step.
#             assert self.norm is not None
#             return self.norm(h)  # type: ignore

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         seqlens: List[int],
#         cache: Optional[BufferCache] = None,
#     ) -> torch.Tensor:
#         h = self.forward_partial(input_ids, seqlens, cache=cache)
#         if self.pipeline_rank < self.num_pipeline_ranks - 1:
#             # ignore the intermediate activations as we'll get the final output from
#             # the last stage
#             outs = torch.empty(
#                 h.shape[0], self.vocab_size, device=h.device, dtype=h.dtype
#             )
#         else:
#             assert self.output is not None
#             outs = self.output(h)
#         if self.num_pipeline_ranks > 1:
#             torch.distributed.broadcast(outs, src=self.num_pipeline_ranks - 1)
#         return outs.float()

#     def load_state_dict(
#         self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
#     ) -> None:
#         state_to_load = {}
#         skipped = set([])
#         for k, v in state_dict.items():
#             if k.startswith("tok_embeddings"):
#                 if self.pipeline_rank == 0:
#                     state_to_load[k] = v
#                 else:
#                     logging.debug(
#                         "Skipping parameter %s at pipeline rank %d",
#                         k,
#                         self.pipeline_rank,
#                     )
#                     skipped.add(k)
#             elif k.startswith("norm") or k.startswith("output"):
#                 if self.pipeline_rank == self.num_pipeline_ranks - 1:
#                     state_to_load[k] = v
#                 else:
#                     logging.debug(
#                         "Skipping parameter %s at pipeline rank %d",
#                         k,
#                         self.pipeline_rank,
#                     )
#                     skipped.add(k)
#             elif k.startswith("layers"):
#                 layer_id = k.split(".")[1]
#                 if layer_id in self.layers:
#                     state_to_load[k] = v
#                 else:
#                     logging.debug(
#                         "Skipping parameter %s at pipeline rank %d",
#                         k,
#                         self.pipeline_rank,
#                     )
#                     skipped.add(k)
#             else:
#                 raise ValueError(f"Unexpected key {k}")
#         assert set(state_dict.keys()) == skipped.union(set(state_to_load.keys()))
#         super().load_state_dict(state_to_load, strict=strict, assign=assign)

#     @staticmethod
#     def from_folder(
#         folder: Union[Path, str],
#         max_batch_size: int = 1,
#         num_pipeline_ranks: int = 1,
#         device: Union[torch.device, str] = "cuda",
#         dtype: Optional[torch.dtype] = None,
#     ) -> "Transformer":
#         with open(Path(folder) / "params.json", "r") as f:
#             model_args = TransformerArgs.from_dict(json.load(f))
#         model_args.max_batch_size = max_batch_size
#         if num_pipeline_ranks > 1:
#             pipeline_rank = torch.distributed.get_rank()
#         else:
#             pipeline_rank = 0
#         with torch.device("meta"):
#             model = Transformer(
#                 model_args,
#                 pipeline_rank=pipeline_rank,
#                 num_pipeline_ranks=num_pipeline_ranks,
#             )

#         pt_model_file = Path(folder) / "consolidated.00.pth"
#         safetensors_model_file = Path(folder) / "consolidated.safetensors"

#         assert (
#             pt_model_file.exists() or safetensors_model_file.exists()
#         ), f"Make sure either {pt_model_file} or {safetensors_model_file} exists"
#         assert not (
#             pt_model_file.exists() and safetensors_model_file.exists()
#         ), f"Both {pt_model_file} and {safetensors_model_file} cannot exist"

#         if pt_model_file.exists():
#             loaded = torch.load(str(pt_model_file), mmap=True)
#         else:
#             loaded = safetensors.torch.load_file(str(safetensors_model_file))

#         model.load_state_dict(loaded, assign=True, strict=True)

#         return model.to(device=device, dtype=dtype)
