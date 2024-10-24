# copied from: https://github.com/Infini-AI-Lab/TriForce/tree/main/utils

import torch
from torch.nn import functional as F

# copied from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """
    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
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


def norm_logits(
    logits: torch.Tensor, temperature=0.6, top_k=-1, top_p=0.9
) -> torch.Tensor:
    """
    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)

    probs = F.softmax(logits, dim=-1)
    return probs


def sample(probs: torch.Tensor, num_samples=1):
    idx_next = torch.multinomial(probs, num_samples=num_samples, replacement=True)
    return idx_next


def max_fn(x):
    """
    norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True)
    if x_max_sum == 0:
        print(x.max(), x.min(), x.shape)
    return x_max / x_max_sum


@torch.inference_mode()
def TriForce(
    tokenizer,
    device,
    input_ids,
    gamma=4,
    max_len=256,
    top_k=-1,
    top_p=0.9,
    temperature=0.6,
):
    # after prefill
    next_token = sample(
        norm_logits(logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p)
    )

    n = 0
    while n < max_len:
        if next_token.shape == torch.Size([1]):
            next_token = next_token.unsqueeze(0)

        # speculative decoding for draft (68m) and retrieval 7b model
        pred_token_idx = next_token
        verify_tokens, speculation_probs, acc_rate_middle = Middle_Spec(
            pred_token_idx, graph_engine, gamma, False, tokenizer
        )
        generated_ids = verify_tokens[1:]

        gamma2 = len(generated_ids)

        # speculative decoding retrieval 7b model and target model
        verify_tokens = torch.cat(
            [
                next_token,
                torch.LongTensor([generated_ids]).to(device),
            ],
            dim=1,
        )
        logits = graph_engine.inference(input_ids=verify_tokens)

        count = 0
        verify_probs = []

        probs = norm_logits(
            logits[0], temperature=temperature, top_k=top_k, top_p=top_p
        )
        for i in range(gamma2 + 1):
            verify_probs.append(probs[i])

        pass_tokens = torch.full(
            (1, gamma2 + 2), 100, device=graph_engine.engine.model.device
        )
        pass_tokens[:, 0] = next_token

        for i, speculation_prob, verify_prob in zip(
            generated_ids, speculation_probs, verify_probs
        ):
            r = torch.rand(1, device=graph_engine.engine.model.device)
            if r < torch.min(
                torch.tensor([1], device=r.device),
                (verify_prob[i] / speculation_prob[i]),
            ):
                count += 1
                accepted_count += 1
                n += 1
                pred_token_idx = torch.tensor([[i]]).to(
                    graph_engine.engine.model.device
                )
                pass_tokens[:, count] = pred_token_idx
                # if eos
                if tokenizer.eos_token_id == i:
                    draft_count -= gamma2 - count
                    break
            else:
                resample_count += 1
                n += 1
                pred_token_idx = sample(max_fn(verify_prob - speculation_prob))
                pass_tokens[:, count + 1] = pred_token_idx
                break

            if tokenizer.eos_token_id == pred_token_idx:
                break

        # update 7b cache
        graph_engine.engine.kv_cache.seq_len -= len(generated_ids) - count
        graph_engine.update_graph_cache()

        if count == len(generated_ids):
            target_sample_count += 1
            n += 1
            pred_token_idx = sample(verify_probs[-1])
            pass_tokens[:, count + 1] = pred_token_idx
            count += 1

        next_token = pred_token_idx
