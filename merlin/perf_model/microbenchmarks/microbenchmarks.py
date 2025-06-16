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
            data[str(i+1)] = round(avg_latencies[i], 3)

    return data


def ceildiv(a, b):
    # from: https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return -(a // -b)


def test_allreduce(model_config: dict, batch_size: int, seq_len: int, group):
    """
    measure the latency of all-reduce in group
    """
    # prepare inputs
    x = torch.rand(
        (batch_size, seq_len, model_config["hidden_size"]),
        dtype=torch.bfloat16,
        device=DEVICE,
    )

    # warmup
    for _ in range(N_WARMUPS):
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)

    # real test
    torch.cuda.synchronize(device=DEVICE)
    tic = time.time()
    for _ in range(N_TESTS):
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
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
    measure the latency of (x @ w1 + x @ w3) @ w2
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
    w1s = [
        torch.rand((interm_d, model_d), dtype=DTYPE, device=DEVICE)
        for _ in range(n_copies)
    ]  # transpose to match the performance of nn.Linear
    w2s = [
        torch.rand((model_d, interm_d), dtype=DTYPE, device=DEVICE)
        for _ in range(n_copies)
    ]
    w3s = [
        torch.rand((interm_d, model_d), dtype=DTYPE, device=DEVICE)
        for _ in range(n_copies)
    ]

    # warm up
    for iter in range(n_warmups):
        i = iter % n_copies
        y = (x @ w1s[i].T + x @ w3s[i].T) @ w2s[i].T

    # real measurement
    torch.cuda.synchronize(device=DEVICE)
    tic = time.time()
    for iter in range(n_tests):
        i = iter % n_copies
        y = (x @ w1s[i].T + x @ w3s[i].T) @ w2s[i].T

    torch.cuda.synchronize(device=DEVICE)
    latency = (time.time() - tic) * 1000 / n_tests

    return latency


def test_qkvo(model_config: dict, tp_size: int, batch_size: int, seq_len: int):
    """
    measure the latency of x @ wq, x @ wk, x @ wv, output @ wo
    """
    model_d = model_config["hidden_size"]
    head_dim = model_d // model_config["num_attention_heads"]
    n_heads = ceildiv(model_config["num_attention_heads"], tp_size)
    n_kv_heads = ceildiv(model_config["num_key_value_heads"], batch_size)
    n_layers = model_config["num_hidden_layers"]

    n_copies = n_layers
    x = torch.rand((batch_size * seq_len, model_d), dtype=DTYPE, device=DEVICE)
    wqs = [
        torch.rand((n_heads * head_dim, model_d), dtype=DTYPE, device=DEVICE)
        for _ in range(n_copies)
    ]
    wks = [
        torch.rand((n_kv_heads * head_dim, model_d), dtype=DTYPE, device=DEVICE)
        for _ in range(n_copies)
    ]
    wvs = [
        torch.rand((n_kv_heads * head_dim, model_d), dtype=DTYPE, device=DEVICE)
        for _ in range(n_copies)
    ]
    wos = [
        torch.rand((model_d, n_heads * head_dim), dtype=DTYPE, device=DEVICE)
        for _ in range(n_copies)
    ]

    # warm up
    for iter in range(N_WARMUPS):
        i = iter % n_copies
        output = x @ wqs[i].T
        y = x @ wks[i].T
        y = x @ wvs[i].T
        y = output @ wos[i].T

    torch.cuda.synchronize(device=DEVICE)
    tic = time.time()
    for iter in range(N_TESTS):
        i = iter % n_copies
        output = x @ wqs[i].T
        y = x @ wks[i].T
        y = x @ wvs[i].T
        y = output @ wos[i].T

    torch.cuda.synchronize(device=DEVICE)
    latency = (time.time() - tic) * 1000 / N_TESTS

    return latency


def test_repeat_kv(model_config: dict, tp_size: int, batch_size: int, seq_len: int):
    """
    measure the latency of repeat_kv
    """
    head_dim = model_config["hidden_size"] // model_config["num_attention_heads"]
    n_layers = model_config["num_hidden_layers"]
    n_kv_heads = ceildiv(model_config["num_key_value_heads"], tp_size)
    n_rep = model_config["num_attention_heads"] // model_config["num_key_value_heads"]

    n_copies = n_layers
    ks = [
        torch.rand(
            (batch_size, n_kv_heads, MAX_SEQ_LEN, head_dim),
            dtype=DTYPE,
            device=DEVICE,
        )
        for _ in range(n_copies)
    ]

    # warm up
    for iter in range(N_WARMUPS):
        i = iter % n_copies
        k = ks[i]
        k = k[:, :, None, :, :].expand(
            batch_size, n_kv_heads, n_rep, MAX_SEQ_LEN, head_dim
        )
        k = k.reshape(batch_size, n_kv_heads * n_rep, MAX_SEQ_LEN, head_dim)

    torch.cuda.synchronize(device=DEVICE)
    tic = time.time()
    for iter in range(N_TESTS):
        i = iter % n_copies
        k = ks[i]
        k = k[:, :, None, :, :].expand(
            batch_size, n_kv_heads, n_rep, MAX_SEQ_LEN, head_dim
        )
        k = k.reshape(batch_size, n_kv_heads * n_rep, MAX_SEQ_LEN, head_dim)
    torch.cuda.synchronize(device=DEVICE)
    latency = (time.time() - tic) * 2 * 1000 / N_TESTS

    return latency


def test_attn_score(model_config: dict, tp_size: int, batch_size: int, seq_len: int):
    """
    measure the latency of Q @ K.T @ V
    """
    n_layers = model_config["num_hidden_layers"]
    n_heads = ceildiv(model_config["num_attention_heads"], tp_size)
    head_dim = model_config["hidden_size"] // model_config["num_attention_heads"]

    n_copies = n_layers
    qs = [
        torch.rand(
            (batch_size, n_heads, seq_len, head_dim),
            dtype=DTYPE,
            device=DEVICE,
        )
        for _ in range(n_copies)
    ]
    ks = [
        torch.rand(
            (batch_size, n_heads, MAX_SEQ_LEN, head_dim),
            dtype=DTYPE,
            device=DEVICE,
        )
        for _ in range(n_copies)
    ]
    vs = [
        torch.rand(
            (batch_size, n_heads, MAX_SEQ_LEN, head_dim),
            dtype=DTYPE,
            device=DEVICE,
        )
        for _ in range(n_copies)
    ]

    # warm up
    for iter in range(N_WARMUPS):
        i = iter % n_copies
        s = qs[i] @ ks[i].transpose(2, 3) @ vs[i]

    torch.cuda.synchronize(device=DEVICE)
    tic = time.time()
    for iter in range(N_TESTS):
        i = iter % n_copies
        s = qs[i] @ ks[i].transpose(2, 3) @ vs[i]
    torch.cuda.synchronize(device=DEVICE)
    latency = (time.time() - tic) * 1000 / N_TESTS

    return latency


def test_router(model_config: dict, tp_size: int, batch_size: int, seq_len: int):
    """
    measure the latency of router
    """
    model_d = model_config["hidden_size"]
    n_layers = model_config["num_hidden_layers"]
    n_experts = model_config["num_local_experts"]

    n_tokens = batch_size * seq_len
    n_copies = n_layers
    x = torch.rand((n_tokens, model_d), dtype=DTYPE, device=DEVICE)
    ws = [
        torch.rand((n_experts, model_d), dtype=DTYPE, device=DEVICE)
        for _ in range(n_copies)
    ]  # transpose to match the performance of nn.Linear

    # warm up
    for iter in range(N_WARMUPS):
        i = iter % n_copies
        y = x @ ws[i].T

    # real measurement
    torch.cuda.synchronize(device=DEVICE)
    tic = time.time()
    for iter in range(N_TESTS):
        i = iter % n_copies
        y = x @ ws[i].T
    torch.cuda.synchronize(device=DEVICE)
    latency = (time.time() - tic) * 1000 / N_TESTS

    return latency


def run_tests(model_config: dict, tp_size, test_func, target_ranks, group):
    avg_latencies = []
    if test_func == test_expert:
        for n_tokens in range(1, END_BSZ*PROMPT_LEN+1):
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
