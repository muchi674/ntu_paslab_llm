import argparse
import json
import os
import time

import torch
import torch.distributed as dist

# Environment variables set by torch.distributed.launch
NODE_RANK = int(os.environ["GROUP_RANK"])
LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])

# general settings
DTYPE = torch.bfloat16
N_WARMUPS = 1000
N_TESTS = 3000
START_BSZ = 1
END_BSZ = 16
MAX_SEQ_LEN = 256
PROMPT_LEN = 128
DEVICE = torch.device(f"cuda:{LOCAL_RANK}")


def format_result(avg_latencies: list[float]):
    shapes = []
    for seq_len in [1, PROMPT_LEN]:
        batch_size = START_BSZ
        while batch_size <= END_BSZ:
            shapes.append(f"{batch_size}-{seq_len}")
            batch_size *= 2

    data = {}

    if len(avg_latencies) == len(shapes):
        for s, l in zip(shapes, avg_latencies):
            data[s] = round(l, 3)
    else:
        for i in range(len(avg_latencies)):
            data[str(i + 1)] = round(avg_latencies[i], 3)

    return data


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return x.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def ceildiv(a, b):
    # from: https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return -(a // -b)


def test_allreduce(model_config: dict, batch_size: int, seq_len: int, group):
    """
    measure the latency of all-reduce in group
    """
    n_allreduce = 10

    # all-reduce function for cuda graph
    def all_reduce_func(inputs):
        for _ in range(n_allreduce):
            dist.all_reduce(inputs, op=dist.ReduceOp.SUM, group=group)

        return inputs

    # prepare inputs
    x = torch.zeros(
        (batch_size, seq_len, model_config["hidden_size"]),
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    dist.barrier(group=group)

    # graph capture
    with torch.cuda.device(device=DEVICE):
        graphed_allreduce = torch.cuda.make_graphed_callables(
            all_reduce_func, (x,), num_warmup_iters=3
        )
    torch.cuda.synchronize(device=DEVICE)

    # warmup
    for _ in range(N_WARMUPS // n_allreduce):
        graphed_allreduce(x)

    # real test
    torch.cuda.synchronize(device=DEVICE)
    tic = time.time()
    for _ in range(N_TESTS // n_allreduce):
        graphed_allreduce(x)
    torch.cuda.synchronize(device=DEVICE)
    latency = (time.time() - tic) * 1000 / N_TESTS  # in ms

    return latency


def test_p2p(model_config: dict, batch_size: int, seq_len: int, target_ranks, group):
    """
    measure the latency of p2p send-recv from sender to receiver
    """
    sender = target_ranks[0]
    receiver = target_ranks[1]

    # prepare inputs
    x = torch.rand(
        (batch_size, seq_len, model_config["hidden_size"]),
        dtype=torch.bfloat16,
        device=DEVICE,
    )

    # warmup
    for _ in range(N_WARMUPS):
        if WORLD_RANK == sender:
            ops = [dist.P2POp(dist.isend, x, receiver)]
        else:
            ops = [dist.P2POp(dist.irecv, x, sender)]
        for req in dist.batch_isend_irecv(ops):
            req.wait()

    # real test
    torch.cuda.synchronize(device=DEVICE)
    tic = time.time()
    for _ in range(N_TESTS):
        if WORLD_RANK == sender:
            ops = [dist.P2POp(dist.isend, x, receiver)]
        else:
            ops = [dist.P2POp(dist.irecv, x, sender)]
        for req in dist.batch_isend_irecv(ops):
            req.wait()

    torch.cuda.synchronize(device=DEVICE)
    latency = (time.time() - tic) * 1000 / N_TESTS
    # dist.barrier(group=group)

    return latency


def test_expert(model_config: dict, tp_size: int, n_tokens: int):
    """
    measure the latency of (silu(x @ w1) * x @ w3) @ w2
    """
    model_d = model_config["hidden_size"]
    interm_d = ceildiv(model_config["intermediate_size"], tp_size)
    n_layers = model_config["num_hidden_layers"]

    n_warmups, n_tests = 100, 300
    if n_tokens > 100:
        n_warmups, n_tests = 1, 5

    # prepare inputs
    n_copies = n_layers  # TODO: how many copies do we need?
    x = torch.rand((n_tokens, model_d), dtype=DTYPE, device=DEVICE)
    w_gate_ups = [
        torch.rand((interm_d * 2, model_d), dtype=DTYPE, device=DEVICE)
        for _ in range(n_copies)
    ]
    w_downs = [
        torch.rand((model_d, interm_d), dtype=DTYPE, device=DEVICE)
        for _ in range(n_copies)
    ]

    # warm up
    for iter in range(n_warmups):
        i = iter % n_copies
        w_gate_up = w_gate_ups[i].T
        w_down = w_downs[i].T
        gate_states, up_states = (x @ w_gate_up).chunk(2, dim=-1)
        hidden_states = torch.nn.functional.silu(gate_states) * up_states
        y = hidden_states @ w_down

    # real measurement
    torch.cuda.synchronize(device=DEVICE)
    tic = time.time()
    for iter in range(n_tests):
        i = iter % n_copies
        w_gate_up = w_gate_ups[i].T
        w_down = w_downs[i].T
        gate_states, up_states = (x @ w_gate_up).chunk(2, dim=-1)
        hidden_states = torch.nn.functional.silu(gate_states) * up_states
        y = hidden_states @ w_down

    torch.cuda.synchronize(device=DEVICE)
    latency = (time.time() - tic) * 1000 / n_tests

    return latency


def test_attn_router(model_config: dict, tp_size: int, batch_size: int, seq_len: int):
    model_d = model_config["hidden_size"]
    head_dim = model_d // model_config["num_attention_heads"]
    n_kv_heads = ceildiv(model_config["num_key_value_heads"], tp_size)
    n_rep = model_config["num_attention_heads"] // model_config["num_key_value_heads"]
    n_heads = n_kv_heads * n_rep
    n_experts = model_config["num_local_experts"]

    wq = torch.rand((n_heads * head_dim, model_d), dtype=DTYPE, device=DEVICE)
    wk = torch.rand((n_kv_heads * head_dim, model_d), dtype=DTYPE, device=DEVICE)
    wv = torch.rand((n_kv_heads * head_dim, model_d), dtype=DTYPE, device=DEVICE)
    wo = torch.rand((model_d, n_heads * head_dim), dtype=DTYPE, device=DEVICE)
    ks = torch.rand(
        (batch_size, n_kv_heads, MAX_SEQ_LEN, head_dim),
        dtype=DTYPE,
        device=DEVICE,
    )
    vs = torch.rand(
        (batch_size, n_kv_heads, MAX_SEQ_LEN, head_dim),
        dtype=DTYPE,
        device=DEVICE,
    )
    wr = torch.rand((n_experts, model_d), dtype=DTYPE, device=DEVICE)

    mask = torch.full(
        (MAX_SEQ_LEN, MAX_SEQ_LEN), float("-inf"), dtype=DTYPE, device=DEVICE
    )
    mask = torch.triu(mask, diagonal=1)
    storage_idx = torch.arange(seq_len, dtype=torch.long, device=DEVICE)

    # graph function
    def attn_func(inputs):
        xq = inputs @ wq.T
        xk = inputs @ wk.T
        xv = inputs @ wv.T

        xq = xq.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        xk = xk.view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(ks, n_rep)
        values = repeat_kv(vs, n_rep)

        output = torch.nn.functional.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=mask[storage_idx],
            dropout_p=0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
        output = output @ wo.T
        output = output @ wr.T
        return output

    x = torch.rand((batch_size * seq_len, model_d), dtype=DTYPE, device=DEVICE)

    # capture graph
    with torch.cuda.device(device=DEVICE):
        graphed_attn = torch.cuda.make_graphed_callables(
            attn_func, (x,), num_warmup_iters=3
        )

    # warmup
    for _ in range(N_WARMUPS):
        graphed_attn(x)

    # real test
    torch.cuda.synchronize(device=DEVICE)
    tic = time.time()
    for _ in range(N_TESTS):
        graphed_attn(x)
    torch.cuda.synchronize(device=DEVICE)
    latency = (time.time() - tic) * 1000 / N_TESTS  # in ms

    return latency


def run_tests(model_config: dict, tp_size, test_func, target_ranks, group):
    avg_latencies = []
    if test_func == test_expert:
        for n_tokens in range(1, END_BSZ * PROMPT_LEN + 1):
            # run microbenchmarks only on target ranks
            if WORLD_RANK in target_ranks:
                latency = test_func(model_config, tp_size, n_tokens)
            else:
                latency = 0.0

            avg_latencies.append(latency)

    else:
        for seq_len in [1, PROMPT_LEN]:
            batch_size = START_BSZ
            while batch_size <= END_BSZ:
                # run microbenchmarks only on target ranks
                if WORLD_RANK in target_ranks:
                    if test_func == test_allreduce:
                        latency = test_func(model_config, batch_size, seq_len, group)
                    elif test_func == test_p2p:
                        latency = test_func(
                            model_config, batch_size, seq_len, target_ranks, group
                        )
                    else:
                        latency = test_func(model_config, tp_size, batch_size, seq_len)
                else:
                    latency = 0.0

                avg_latencies.append(latency)  # latency is in ms
                batch_size *= 2

    dist.barrier()

    # broadcast result to all ranks
    avg_latencies_tensor = torch.tensor(
        avg_latencies, dtype=torch.float32, device=DEVICE
    )
    dist.broadcast(avg_latencies_tensor, target_ranks[0])
    avg_latencies = avg_latencies_tensor.tolist()

    return format_result(avg_latencies)
