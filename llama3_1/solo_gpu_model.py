import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from llama_models.llama3.api.tokenizer import Tokenizer

DEFAULT_SEED = 7
MAX_BATCH_SIZE = 32
MAX_SEQ_LEN = 512


def get_json(file_path: Path) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


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
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False

    max_batch_size: int = 32
    max_seq_len: int = 2048

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, device: torch.device):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        self.n_kv_heads = self.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            ),
            device=device,
        )
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_kv_heads,
                self.head_dim,
            ),
            device=device,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, device: torch.device):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, device)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, device: torch.device):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params, device))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output


class Llama3_1:

    @staticmethod
    def build(model_path: str, device: torch.device) -> "Llama3_1":
        assert torch.cuda.is_bf16_supported()
        torch.set_default_dtype(torch.bfloat16)
        torch.manual_seed(DEFAULT_SEED)
        start_time = time.time()

        ckpt_dir = Path(model_path)
        checkpoints = sorted(ckpt_dir.glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"

        state_dict = {}
        for ckpt_path in checkpoints:
            state_dict.update(
                torch.load(
                    ckpt_path,
                    map_location=device,
                    weights_only=True,
                    mmap=True,
                )
            )

        params = get_json(ckpt_dir / "params.json")
        model_args: ModelArgs = ModelArgs(
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=MAX_BATCH_SIZE,
            **params,
        )
        tokenizer = Tokenizer(model_path=str(ckpt_dir / "tokenizer.model"))

        assert (
            model_args.vocab_size == tokenizer.n_words
        ), f"model_args vocab = {model_args.vocab_size} but tokenizer vocab = {tokenizer.n_words}"

        with torch.device("meta"):
            # device here specifies where to pre-allocate KV-cache
            model = Transformer(model_args, device=device)
        model.load_state_dict(state_dict, assign=True, strict=True)

        print(f"Loaded model in {time.time() - start_time:.2f} seconds")
        return Llama3_1(model, tokenizer, model_args)

    def __init__(self, model: Transformer, tokenizer: Tokenizer, args: ModelArgs):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        max_gen_len: int,
        cpu: torch.device,
        gpu: torch.device,
        temperature: float = 0.0,
        top_p: float = 0.0,
    ) -> tuple:
        assert max_gen_len > 0

        model = self.model.eval()
        params = model.params
        bsz = len(prompts)

        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

        encoded_prompts: list[list[int]] = [
            self.tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts
        ]
        min_prompt_len = min(len(p) for p in encoded_prompts)
        max_prompt_len = max(len(p) for p in encoded_prompts)

        if max_prompt_len + max_gen_len >= params.max_seq_len:
            print(
                f"Out of token budget {max_prompt_len + max_gen_len} vs {params.max_seq_len}"
            )
            return

        total_len = max_gen_len + max_prompt_len
        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=cpu)
        for k, t in enumerate(encoded_prompts):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=cpu)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=cpu)
        input_text_mask = tokens != pad_id

        # TODO: not sure what this is for
        if min_prompt_len == total_len:
            logits = model.forward(tokens, prev_pos)

        stop_tokens = torch.tensor(self.tokenizer.stop_tokens)
        gen_tokens = []

        # notice:
        # 1. it seems that prompts with length < max will generate
        # total_len - len(prompt) tokens
        # 2. when batch size > 1, only the first bsz * min_prompt_len tokens
        # will be processed in parallel. Longer prompts' remaining tokens are
        # evaluated one-by-one with the min prompt's token generation
        for cur_pos in range(min_prompt_len, total_len):
            logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            flat_next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            flat_next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], flat_next_token
            )
            tokens[:, cur_pos] = flat_next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(flat_next_token, stop_tokens)
            )

            gen_tokens.append(next_token)
            prev_pos = cur_pos
            if all(eos_reached):
                break

        gen_tokens = torch.cat(gen_tokens, 1).tolist()
        n_gen_tkns = sum(len(y) - 1 for y in gen_tokens)
        responses = [self.tokenizer.decode(y) for y in gen_tokens]

        end_time.record()
        torch.cuda.synchronize(device=gpu)
        total_latency = start_time.elapsed_time(end_time) / 1000  # to seconds

        return (
            sum(len(p) for p in encoded_prompts),
            n_gen_tkns,
            responses,
            total_latency,
        )


def main(
    model_path: str,
    prompt: str,
    prompt_path: str,
    n_prompts: int,
    max_gen_len: int,
    hide_resp: bool,
):
    assert prompt or (prompt_path and n_prompts and n_prompts > 0)

    cpu = torch.device("cpu")
    gpu_0 = torch.device("cuda:0")
    prompts: list[str] = None
    if prompt:
        prompts = [prompt]
    else:
        dataset: list[str] = get_json(Path(prompt_path))["prompts"]
        n_repeats = -(n_prompts // -len(dataset))  # ceil division
        prompts = (dataset * n_repeats)[:n_prompts]
    model = Llama3_1.build(model_path, cpu)

    # warmup
    model.generate(prompts=["hello, how are you?"], max_gen_len=1, gpu=gpu_0)
    print("finished warming up")

    n_p_tkns, n_gen_tkns, responses, total_latency = model.generate(
        prompts=prompts, max_gen_len=max_gen_len, gpu=gpu_0
    )

    print("=" * 20)
    print("PERFORMANCE BREAKDOWN\n")
    print(f"num prompt tokens: {n_p_tkns}")
    print(f"num generated tokens: {n_gen_tkns}")
    print(f"total latency: {total_latency:.2f} secs")
    print(f"mixed throughput: {((n_p_tkns + n_gen_tkns) / total_latency):.2f} t/s")
    if not hide_resp:
        print("=" * 20)
        print("In-n-Outs")
        for p, resp in zip(prompts, responses):
            print(f"PROMPT:\n{p}")
            print(f"RESPONSE:\n{resp}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--prompt-path", type=str)
    parser.add_argument("--n-prompts", type=int)
    parser.add_argument("--max-gen-len", type=int)
    parser.add_argument("--hide-resp", action="store_true")
    args = parser.parse_args()

    main(
        args.model_path,
        args.prompt,
        args.prompt_path,
        args.n_prompts,
        args.max_gen_len,
        args.hide_resp,
    )