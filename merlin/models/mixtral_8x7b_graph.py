from dataclasses import dataclass
from pathlib import Path
from statistics import mean
import argparse
import json
import os
import time

from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

# Environment variables set by torch.distributed.launch
LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
GROUP_RANK = int(os.environ["GROUP_RANK"])
WORLD_RANK = int(os.environ["RANK"])

DEFAULT_SEED = 7


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
    freqs_cis = freqs_cis[None, None, :, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return x.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


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
    parallel_attn: bool = False

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
        self.freqs_cis: torch.Tensor
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
        freqs_cis: torch.Tensor,
        cache: torch.Tensor,
        mask: torch.Tensor,
        prefill_storage_idx: torch.Tensor,
        decode_storage_idx: torch.Tensor,
    ):
        self.freqs_cis = freqs_cis
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
        xq, xk = apply_rotary_emb(xq, xk, self.freqs_cis[storage_idx])

        # assumes bsz matches that of cache
        self.cache[0, self.li].index_copy_(dim=-2, index=storage_idx, source=xk)
        self.cache[1, self.li].index_copy_(dim=-2, index=storage_idx, source=xv)
        keys = self.cache[0, self.li]
        values = self.cache[1, self.li]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.repeats)  # (bs, max_seq_len, n_heads, head_dim)
        values = repeat_kv(values, self.repeats)  # (bs, max_seq_len, n_heads, head_dim)

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
        gate_logits = self.gate(x)
        topk_weight, topk_ids = torch.topk(gate_logits, self.num_experts_per_tok)
        topk_weight = F.softmax(topk_weight, dim=1, dtype=torch.float).to(x.dtype)
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
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


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
        # WARNING: assumes attention is intra-node TP
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
        # WARNING: assumes attention is intra-node TP
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
        self._precomputed_freqs_cis: torch.Tensor = None
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
        freqs_cis: torch.Tensor,
        cache: torch.Tensor,
        mask: torch.Tensor,
        prefill_storage_idx: torch.Tensor,
        decode_storage_idx: torch.Tensor,
    ):
        for li in range(self.args.first_layer, self.args.last_layer + 1):
            self.layers[str(li)].attention.set_batch_level_args(
                freqs_cis, cache, mask, prefill_storage_idx, decode_storage_idx
            )
        if self.args.has_pp and not self.args.is_first_stage:
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
        if self.args.has_pp and not self.args.is_last_stage:
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
            if self.args.parallel_attn:
                idx += 1
            return options[idx]

        def select_subseq_graphable(prefill: bool, options: tuple):
            idx = 0
            if not prefill:
                idx += len(options) // 2
            if self.args.parallel_attn:
                idx += 3
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
                self.layers[k].first_prefill_graphable_intra_attn,
                self.layers[k].first_decode_graphable,
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
                    self.layers[k].subseq_prefill_graphable_intra_attn_inter_moe,
                    self.layers[k].subseq_prefill_graphable_intra_attn_intra_moe,
                    self.layers[k].subseq_decode_graphable,
                    self.layers[k].subseq_decode_graphable_inter_moe,
                    self.layers[k].subseq_decode_graphable_intra_moe,
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
        non_experts_filename = "non-experts.pt"
        if not (model_path / non_experts_filename).is_file():
            non_experts_filename = f"non-experts-{node_id}-{LOCAL_RANK}.pt"
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

        # check if intra-node TP is applied on attention
        if (
            non_experts[f"layers.{model_args.first_layer}.attention.wq.weight"].shape[0]
            < model_args.n_heads * model_args.head_dim
        ):
            assert model_args.n_heads % LOCAL_WORLD_SIZE == 0
            assert model_args.n_kv_heads % LOCAL_WORLD_SIZE == 0
            model_args.n_heads //= LOCAL_WORLD_SIZE
            model_args.n_kv_heads //= LOCAL_WORLD_SIZE
            model_args.parallel_attn = True

        comms: list
        if (
            model_args.has_pp and model_args.parallel_experts
        ) or model_args.parallel_attn:
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
        tokenizer = MistralTokenizer.v1()

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
        freqs_cis = precompute_freqs_cis(
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
            freqs_cis,
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
        dist.barrier()
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

        dist.barrier()
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
            # dist.barrier()
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

            if prev_pos == 0:
                prefill_time = time.time() - tic
                tic = time.time()
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

            prev_pos = cur_pos
            if all(eos_reached):
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
    start = 0
    for end in range(batch_size, n_prompts + 1, batch_size):
        prompt_batch = prompts[start:end]
        bsz = len(prompt_batch)
        responses, n_p_tkns, n_gen_tkns, prefill_time, decode_time = model.generate(
            prompt_batch,
            max_gen_len=max_gen_len,
            temperature=0.0,
            device=gpu,
            profile=end == n_prompts,
        )

        if WORLD_RANK == 0:
            prefill_tp = n_p_tkns / prefill_time
            decode_tp = n_gen_tkns / decode_time
            if n_gen_tkns / bsz > max_gen_len * 0.9:
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

        start = end
        time.sleep(3)

    if WORLD_RANK == 0:
        print("=" * 20)
        print("RUN STATISTICS")
        print(f"avg prefill throughput: {mean(prefill_tps):.2f} t/s")
        print(f"avg decode throughput: {mean(decode_tps):.2f} t/s")

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
