from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import json
import os

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
) -> Tuple[torch.Tensor, torch.Tensor]:
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


class MoeLayer(nn.Module):
    def __init__(self, args: ModelArgs, gate: nn.Module):
        super().__init__()
        self.num_experts: int = args.moe["num_experts"]
        self.num_experts_per_tok: int = args.moe["num_experts_per_tok"]
        self.gate = gate

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=-1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)

        # selected_experts = selected_experts.to("cpu")
        # eis, bis, nes = [], [], []
        # for ei in range(self.num_experts):
        #     batch_idx, nth_expert = torch.where(selected_experts == ei)
        #     if torch.numel(batch_idx) > 0:
        #         eis.append(ei)
        #         bis.append(batch_idx.to(device=inputs.device))
        #         nes.append(nth_expert.to(device=inputs.device))

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
    def __init__(self, args: ModelArgs, li: int):
        super().__init__()
        self.attention = Attention(args, li)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.feed_forward = MoeLayer(
            args=args,
            gate=nn.Linear(args.dim, args.moe["num_experts"], bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: torch.Tensor,
        mask: torch.Tensor,
        storage_idx: torch.Tensor,
    ) -> torch.Tensor:
        r = self.attention(self.attention_norm(x), freqs_cis, cache, mask, storage_idx)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args: ModelArgs = args
        self.freqs_cis: torch.Tensor = None
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.layers = nn.ModuleDict(
            {str(li): TransformerBlock(args, li) for li in range(args.n_layers)}
        )
        self.prefill_graphed_callables: list = []
        self.decode_graphed_callables: list = []

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
        prefill_graphed_callables, decode_graphed_callables = [], []
        with torch.cuda.device(device=self.device):
            for li in range(self.args.n_layers):
                prefill_graphed_callables.append(
                    torch.cuda.make_graphed_callables(
                        self.layers[str(li)].forward,
                        (prefill_x, self.freqs_cis, cache, mask, prefill_storage_x),
                        num_warmup_iters=128,
                    )
                )
                decode_graphed_callables.append(
                    torch.cuda.make_graphed_callables(
                        self.layers[str(li)].forward,
                        (decode_x, self.freqs_cis, cache, mask, decode_storage_x),
                        num_warmup_iters=128,
                    )
                )
        self.prefill_graphed_callables = prefill_graphed_callables
        self.decode_graphed_callables = decode_graphed_callables

    def forward(
        self,
        tokens: torch.Tensor,
        cache: torch.Tensor,
        mask: torch.Tensor,
        storage_idx: torch.Tensor,
    ):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        for li in range(self.args.n_layers):
            dist.barrier()
            if seqlen > 1:
                layer = self.prefill_graphed_callables[li]
            else:
                layer = self.decode_graphed_callables[li]
            h = layer(h, self.freqs_cis, cache, mask, storage_idx)
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

        with torch.device("meta"):
            model = Transformer(args=model_args)
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

    def get_mask(self, max_seq_len: int, dtype: torch.dtype, device: torch.device):
        mask = torch.full(
            (max_seq_len, max_seq_len), float("-inf"), dtype=dtype, device=device
        )
        mask = torch.triu(mask, diagonal=1)
        return mask

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        *,
        max_batch_size: int,
        max_gen_len: int,
        temperature: float,
        device: torch.device,
        profile: bool = False,
    ) -> Tuple[List[str], int, float, int, float]:

        encoded_prompts = self.encode_prompts(prompts)
        min_p_len = min(len(p) for p in encoded_prompts)
        max_p_len = max(len(p) for p in encoded_prompts)
        max_seq_len = max_p_len + max_gen_len
        bsz = len(encoded_prompts)

        model = self.model.eval()
        cache = self.get_cache(max_batch_size, max_seq_len, device)
        mask = self.get_mask(max_seq_len, model.dtype, device)
        model.draw_graphs(max_batch_size, min_p_len, cache, mask)

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

        if profile:
            torch.cuda.cudart().cudaProfilerStop()

        return n_p_tkns, n_gen_tkns, responses


def main(
    model_path: str,
    prompt: str,
    prompt_path: str,
    n_prompts: int = 1,
    batch_size: int = 1,
    max_gen_len: int = 128,
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

    start = 0
    for end in range(batch_size, n_prompts + 1, batch_size):
        prompt_batch = prompts[start:end]
        model.generate(
            prompt_batch,
            max_batch_size=len(prompt_batch),
            max_gen_len=max_gen_len,
            temperature=0.0,
            device=gpu,
            profile=True,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    MODEL_PATH = "/mnt/llm_team/merlin_mixtral_weights/v0"
    PROMPT_PATH = "/home/muchichen/ntu_paslab_llm/mixtral/prompts/diverse_short.json"
    N_PROMPTS = 4
    BATCH_SIZE = 1
    MAX_GEN_LEN = 16
    main(
        model_path=MODEL_PATH,
        prompt=None,
        prompt_path=PROMPT_PATH,
        n_prompts=N_PROMPTS,
        batch_size=BATCH_SIZE,
        max_gen_len=MAX_GEN_LEN,
    )
    # nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --gpu-metrics-devices=all --gpuctxsw=true torchrun --nnodes=1 --node-rank=0 --nproc-per-node=2 --master-addr=10.10.10.1 --master-port=9091 graph_attn_gate.py
