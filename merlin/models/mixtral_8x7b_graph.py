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
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
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
    freqs_cis = freqs_cis[None, :, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


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

        self.n_heads: int = args.n_heads
        self.head_dim: int = args.head_dim
        self.sqrt_head_dim = self.head_dim**-0.5
        self.n_kv_heads: int = args.n_kv_heads
        self.repeats = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: torch.Tensor,
        mask: torch.Tensor,
        storage_idx: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis[storage_idx])

        # assumes bsz matches that of cache
        cache[0, self.li].index_copy(dim=-3, index=storage_idx, source=xk)
        cache[1, self.li].index_copy(dim=-3, index=storage_idx, source=xv)
        keys = cache[0, self.li]
        values = cache[1, self.li]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.repeats)  # (bs, max_seq_len, n_heads, head_dim)
        values = repeat_kv(values, self.repeats)  # (bs, max_seq_len, n_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_heads, max_seq_len, head_dim)
        values = values.transpose(1, 2)  # (bs, n_heads, max_seq_len, head_dim)

        # (bs, n_heads, seqlen, max_seq_len)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / self.sqrt_head_dim
        scores = scores + mask[storage_idx]
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        output = torch.matmul(scores, values)  # (bs, n_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class Experts:

    def __init__(self, ws: dict):
        self.ws: dict[str, torch.Tensor] = ws

    def forward(self, li: int, ei: int, x: torch.Tensor) -> torch.Tensor:
        w1: torch.Tensor = self.ws[f"{li}.{ei}.w1"].T
        w2: torch.Tensor = self.ws[f"{li}.{ei}.w2"]
        w3: torch.Tensor = self.ws[f"{li}.{ei}.w3"].T
        return (nn.functional.silu(x @ w1) * (x @ w3)) @ w2


class MoeLayer(nn.Module):
    def __init__(self, args: ModelArgs, li: int, gate: nn.Module, experts: Experts):
        super().__init__()
        self.num_experts: int = args.moe["num_experts"]
        self.num_experts_per_tok: int = args.moe["num_experts_per_tok"]
        self.li = li
        self.gate = gate
        self.experts = experts

    def router(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # WARNING: assumes x to be 2D: (batch_size * seq_len, model_dim)
        gate_logits = self.gate(x)
        topk_weight, topk_idx = torch.topk(gate_logits, self.num_experts_per_tok)
        topk_weight = F.softmax(topk_weight, dim=1, dtype=torch.float).to(x.dtype)
        return topk_idx, topk_weight

    def experts_infer(self, x, topk_ids, topk_weight):
        # WARNING: assumes x to be 2D: (batch_size * seq_len, model_dim)
        cnts = topk_ids.new_zeros((topk_ids.shape[0], 8))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0).cpu().numpy()
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            if num_tokens == 0:
                continue
            end_idx = start_idx + num_tokens
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = self.experts.forward(self.li, i, tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


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
        self.attention = Attention(args, li)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.feed_forward = MoeLayer(
            args=args,
            li=li,
            gate=nn.Linear(args.dim, args.moe["num_experts"], bias=False),
            experts=experts,
        )
        self.prefill_graphed_half = None
        self.decode_graphed_half = None

    def run_graphable_half(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: torch.Tensor,
        mask: torch.Tensor,
        storage_idx: torch.Tensor,
    ):
        r = self.attention(self.attention_norm(x), freqs_cis, cache, mask, storage_idx)
        h = x + r  # (batch_size, seq_len, model_dim)
        r = self.ffn_norm(h).view(-1, h.shape[-1])  # (batch_size * seq_len, model_dim)
        topk_idx, topk_weight = self.feed_forward.router(r)
        return h, r, topk_idx, topk_weight

    def forward(
        self,
        x: torch.Tensor,  # (batch_size, seq_len, model_dim)
        freqs_cis: torch.Tensor,
        cache: torch.Tensor,
        mask: torch.Tensor,
        storage_idx: torch.Tensor,
    ) -> torch.Tensor:
        graphed_half = (
            self.prefill_graphed_half if x.shape[1] > 1 else self.decode_graphed_half
        )
        # h.shape = (batch_size, seq_len, model_dim)
        # r.shape = (batch_size * seq_len, model_dim)
        h, r, topk_idx, topk_weight = graphed_half(
            x, freqs_cis, cache, mask, storage_idx
        )
        r = self.feed_forward.experts_infer(r, topk_idx, topk_weight).view(h.shape)
        dist.all_reduce(r, op=dist.ReduceOp.SUM)
        return h + r


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, experts: Experts):
        super().__init__()
        self.args: ModelArgs = args
        self.freqs_cis: torch.Tensor = None
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.layers = nn.ModuleDict(
            {
                str(li): TransformerBlock(args, li, experts)
                for li in range(args.n_layers)
            }
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def draw_graphs(
        self,
        batch_size: int,
        prefill_len: int,
        cache: torch.Tensor,
        mask: torch.Tensor,
    ):
        prefill_x = torch.ones(
            (batch_size, prefill_len, self.args.dim),
            dtype=self.dtype,
            device=self.device,
        )
        decode_x = torch.ones(
            (batch_size, 1, self.args.dim), dtype=self.dtype, device=self.device
        )
        prefill_storage_x = torch.arange(
            0, prefill_len, dtype=torch.long, device=self.device
        )
        decode_storage_x = torch.arange(
            prefill_len, prefill_len + 1, dtype=torch.long, device=self.device
        )
        with torch.cuda.device(device=self.device):
            for li in range(self.args.n_layers):
                layer = self.layers[str(li)]
                layer.prefill_graphed_half = torch.cuda.make_graphed_callables(
                    layer.run_graphable_half,
                    (prefill_x, self.freqs_cis, cache, mask, prefill_storage_x),
                    num_warmup_iters=128,
                )
                layer.decode_graphed_half = torch.cuda.make_graphed_callables(
                    layer.run_graphable_half,
                    (decode_x, self.freqs_cis, cache, mask, decode_storage_x),
                    num_warmup_iters=128,
                )

    def forward(
        self,
        tokens: torch.Tensor,
        cache: torch.Tensor,
        mask: torch.Tensor,
        storage_idx: torch.Tensor,
    ):
        h = self.tok_embeddings(tokens)
        for li in range(self.args.n_layers):
            h = self.layers[str(li)].forward(
                h, self.freqs_cis, cache, mask, storage_idx
            )
        return self.output(self.norm(h)).float()


class Mixtral8x7B:

    @staticmethod
    def build(model_path: str, device: torch.device) -> "Mixtral8x7B":
        torch.manual_seed(DEFAULT_SEED)

        model_path = Path(model_path)
        model_args = ModelArgs.from_hf_config(get_json(model_path / "config.json"))
        non_experts = torch.load(
            model_path / "non-experts.pt",
            map_location=device,
            weights_only=True,
            mmap=True,
        )
        experts = torch.load(
            model_path / f"experts-{WORLD_RANK}.pt",
            map_location=device,
            weights_only=True,
            mmap=True,
        )

        with torch.device("meta"):
            model = Transformer(args=model_args, experts=Experts(experts))
        model.load_state_dict(non_experts, assign=True, strict=True)
        model.freqs_cis = precompute_freqs_cis(
            dim=model_args.head_dim,
            end=128_000,
            theta=model_args.rope_theta,
            device=device,
        )
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
                self.model.args.n_layers,
                max_batch_size,
                max_seq_len,
                self.model.args.n_kv_heads,
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
        max_batch_size: int,
        max_gen_len: int,
        temperature: float,
        device: torch.device,
        draw_new_graph: bool = True,
        profile: bool = False,
    ) -> tuple[list[str], int, float, int, float]:

        encoded_prompts = self.encode_prompts(prompts)
        min_p_len = min(len(p) for p in encoded_prompts)
        max_p_len = max(len(p) for p in encoded_prompts)
        max_seq_len = max_p_len + max_gen_len
        bsz = len(encoded_prompts)

        model = self.model.eval()
        cache = self.get_cache(max_batch_size, max_seq_len, device)
        mask = self.get_mask(max_seq_len, model.dtype, device)
        if draw_new_graph:
            model.draw_graphs(max_batch_size, min_p_len, cache, mask)
            self.clear_cache(cache)
        dist.barrier()

        tic = time.time()
        prefill_time: float  # in sec
        decode_time: float  # in sec
        if profile:
            torch.cuda.cudart().cudaProfilerStart()

        pad_id = max(tkn for p in encoded_prompts for tkn in p) + 1
        tokens = torch.full((bsz, max_seq_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(encoded_prompts):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        prev_pos = 0
        eos_id = self.tokenizer.instruct_tokenizer.tokenizer.eos_id
        eos_reached = torch.tensor([False] * bsz, device=device)
        input_text_mask = tokens != pad_id

        # notice:
        # 1. it seems that prompts with length < max will generate
        # max_seq_len - len(prompt) tokens
        # 2. when batch size > 1, only the first bsz * min_prompt_len tokens
        # will be processed in parallel. Longer prompts' remaining tokens are
        # evaluated one-by-one with the min prompt's token generation
        for cur_pos in range(min_p_len, max_seq_len):
            storage_idx = torch.arange(
                prev_pos, cur_pos, dtype=torch.long, device=device
            )
            logits = model.forward(
                tokens[:, prev_pos:cur_pos], cache, mask, storage_idx
            )

            if cur_pos == min_p_len:
                prefill_time = time.time() - tic
                tic = time.time()
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, 0.8)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & next_token == eos_id

            prev_pos = cur_pos
            if all(eos_reached):
                break

        # this part is from here:
        # https://github.com/meta-llama/llama3/blob/main/llama/generation.py
        responses = []
        n_p_tkns, n_gen_tkns = 0, 0
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
            n_p_tkns += p_len
            n_gen_tkns += len(tkns)

        decode_time = time.time() - tic
        if profile:
            torch.cuda.cudart().cudaProfilerStop()

        return responses, n_p_tkns, n_gen_tkns, prefill_time, decode_time


def main(
    model_path: str,
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
    model = Mixtral8x7B.build(model_path, gpu)

    # warmup
    model.generate(
        ["hello, how are you?"],
        max_batch_size=1,
        max_gen_len=128,
        temperature=0.0,
        device=gpu,
    )
    print("finished warming up")

    prefill_tps = []
    decode_tps = []
    start = 0
    for end in range(batch_size, n_prompts + 1, batch_size):
        prompt_batch = prompts[start:end]
        responses, n_p_tkns, n_gen_tkns, prefill_time, decode_time = model.generate(
            prompt_batch,
            max_batch_size=len(prompt_batch),
            max_gen_len=max_gen_len,
            temperature=0.0,
            device=gpu,
            draw_new_graph=start == 0,
            profile=True,
        )

        if WORLD_RANK == 0:
            prefill_tp = n_p_tkns / prefill_time
            decode_tp = n_gen_tkns / decode_time
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
                print(f"AVG seqlen: {(n_p_tkns / len(prompt_batch)):2f}")
                for p, resp in zip(prompt_batch, responses):
                    print(f"PROMPT:\n{p}")
                    print(f"RESPONSE:\n{resp}\n")

    if WORLD_RANK == 0:
        print("=" * 20)
        print("RUN STATISTICS")
        print(f"avg prefill throughput: {mean(prefill_tps):.2f} t/s")
        print(f"avg decode throughput: {mean(decode_tps):.2f} t/s")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--node-id", type=int)  # ignored
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--prompt-path", type=str)
    parser.add_argument("--n-prompts", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
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

    # nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --gpu-metrics-devices=all --gpuctxsw=true torchrun --nnodes=1 --node-rank=0 --nproc-per-node=2 --master-addr=10.10.10.1 --master-port=9091 graph_attn_gate.py
