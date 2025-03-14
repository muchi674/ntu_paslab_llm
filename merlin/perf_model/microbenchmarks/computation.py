import argparse
import json
import time

import torch

# general settings
DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
N_WARMUPS = 5
N_SAMPLES = 100


def ceildiv(a, b):
    # from: https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return -(a // -b)


def test_expert(model_config: dict, params: dict):
    """
    measure the latency of (x @ w1 + x @ w3) @ w2
    """
    model_d = model_config["hidden_size"]
    interm_d = ceildiv(model_config["intermediate_size"], params["tp_size"])
    n_layers = model_config["num_hidden_layers"]
    n_experts = model_config["num_local_experts"]
    top_k = model_config["num_experts_per_tok"]

    # prepare inputs
    n_tokens = max(params["batch_size"] * params["seq_len"] * top_k // n_experts, 1)
    n_copies = n_layers  # TODO: how many copies do we need?
    x = torch.rand((n_tokens, model_d), dtype=DTYPE, device=DEVICE)
    w1s = [
        torch.rand((interm_d, model_d), dtype=DTYPE, device=DEVICE)
        for _ in range(n_copies)
    ]
    w2s = [
        torch.rand((model_d, interm_d), dtype=DTYPE, device=DEVICE)
        for _ in range(n_copies)
    ]
    w3s = [
        torch.rand((interm_d, model_d), dtype=DTYPE, device=DEVICE)
        for _ in range(n_copies)
    ]

    # warm up
    for _ in range(N_WARMUPS):
        for i in range(n_copies):
            y = (x @ w1s[i].T + x @ w3s[i].T) @ w2s[i].T

    # real measurement
    torch.cuda.synchronize(device=DEVICE)
    tic = time.time()
    for _ in range(N_SAMPLES):
        for i in range(n_copies):
            y = (x @ w1s[i].T + x @ w3s[i].T) @ w2s[i].T

    torch.cuda.synchronize(device=DEVICE)
    latency = (time.time() - tic) * 1000 / (N_SAMPLES * n_copies)

    return latency


def test_qkvo(model_config: dict, params: dict):
    """
    measure the latency of x @ wq, x @ wk, x @ wv, output @ wo
    """
    model_d = model_config["hidden_size"]
    head_dim = model_d // model_config["num_attention_heads"]
    n_heads = ceildiv(model_config["num_attention_heads"], params["tp_size"])
    n_kv_heads = ceildiv(model_config["num_key_value_heads"], params["tp_size"])
    n_layers = model_config["num_hidden_layers"]

    n_copies = n_layers
    x = torch.rand(
        (params["batch_size"] * params["seq_len"], model_d), dtype=DTYPE, device=DEVICE
    )
    wqs = [
        torch.rand((n_heads * head_dim, model_d), dtype=DTYPE, device=DEVICE)
        for _ in range(n_copies)
    ]  # transpose to match the performance of nn.Linear
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
    for _ in range(N_WARMUPS):
        for i in range(n_copies):
            output = x @ wqs[i].T
            y = x @ wks[i].T
            y = x @ wvs[i].T
            y = output @ wos[i].T

    torch.cuda.synchronize(device=DEVICE)
    tic = time.time()
    for _ in range(N_SAMPLES):
        for i in range(n_copies):
            output = x @ wqs[i].T
            y = x @ wks[i].T
            y = x @ wvs[i].T
            y = output @ wos[i].T

    torch.cuda.synchronize(device=DEVICE)
    latency = (time.time() - tic) * 1000 / (N_SAMPLES * n_copies)

    return latency


def test_repeat_kv(model_config: dict, params: dict):
    """
    measure the latency of repeat_kv
    """
    head_dim = model_config["hidden_size"] // model_config["num_attention_heads"]
    n_layers = model_config["num_hidden_layers"]
    n_kv_heads = ceildiv(model_config["num_key_value_heads"], params["tp_size"])
    n_rep = model_config["num_attention_heads"] // model_config["num_key_value_heads"]

    max_seq_len = params["max_seq_len"]
    bs = params["batch_size"]
    n_copies = n_layers
    ks = [
        torch.rand(
            (bs, n_kv_heads, max_seq_len, head_dim),
            dtype=DTYPE,
            device=DEVICE,
        )
        for _ in range(n_copies)
    ]

    # warm up
    for _ in range(N_WARMUPS):
        for k in ks:
            k = k[:, :, None, :, :].expand(bs, n_kv_heads, n_rep, max_seq_len, head_dim)
            k = k.reshape(bs, n_kv_heads * n_rep, max_seq_len, head_dim)

    torch.cuda.synchronize(device=DEVICE)
    tic = time.time()
    for _ in range(N_SAMPLES):
        for k in ks:
            k = k[:, :, None, :, :].expand(bs, n_kv_heads, n_rep, max_seq_len, head_dim)
            k = k.reshape(bs, n_kv_heads * n_rep, max_seq_len, head_dim)
    torch.cuda.synchronize(device=DEVICE)
    latency = (time.time() - tic) * 2 * 1000 / (N_SAMPLES * n_copies)

    return latency


def test_attn_score(model_config: dict, params: dict):
    """
    measure the latency of Q @ K.T @ V
    """
    n_layers = model_config["num_hidden_layers"]
    n_heads = ceildiv(model_config["num_attention_heads"], params["tp_size"])
    head_dim = model_config["hidden_size"] // model_config["num_attention_heads"]

    n_copies = n_layers
    qs = [
        torch.rand(
            (params["batch_size"], n_heads, params["seq_len"], head_dim),
            dtype=DTYPE,
            device=DEVICE,
        )
        for _ in range(n_copies)
    ]
    ks = [
        torch.rand(
            (params["batch_size"], n_heads, params["max_seq_len"], head_dim),
            dtype=DTYPE,
            device=DEVICE,
        )
        for _ in range(n_copies)
    ]
    vs = [
        torch.rand(
            (params["batch_size"], n_heads, params["max_seq_len"], head_dim),
            dtype=DTYPE,
            device=DEVICE,
        )
        for _ in range(n_copies)
    ]

    # warm up
    for _ in range(N_WARMUPS):
        for i in range(n_copies):
            s = qs[i] @ ks[i].transpose(2, 3) @ vs[i]

    torch.cuda.synchronize(device=DEVICE)
    tic = time.time()
    for _ in range(N_SAMPLES):
        for i in range(n_copies):
            s = qs[i] @ ks[i].transpose(2, 3) @ vs[i]
    torch.cuda.synchronize(device=DEVICE)
    latency = (time.time() - tic) * 1000 / (N_SAMPLES * n_copies)

    return latency


def test_router(model_config: dict, params: dict):
    """
    measure the latency of router
    """
    model_d = model_config["hidden_size"]
    n_layers = model_config["num_hidden_layers"]
    n_experts = model_config["num_local_experts"]

    n_tokens = params["batch_size"] * params["seq_len"]
    n_copies = n_layers
    x = torch.rand((n_tokens, model_d), dtype=DTYPE, device=DEVICE)
    ws = [
        torch.rand((n_experts, model_d), dtype=DTYPE, device=DEVICE)
        for _ in range(n_copies)
    ]  # transpose to match the performance of nn.Linear

    # warm up
    for _ in range(N_WARMUPS):
        for w in ws:
            y = x @ w.T

    # real measurement
    torch.cuda.synchronize(device=DEVICE)
    tic = time.time()
    for _ in range(N_SAMPLES):
        for w in ws:
            y = x @ w.T
    torch.cuda.synchronize(device=DEVICE)
    latency = (time.time() - tic) * 1000 / (N_SAMPLES * n_copies)

    return latency


def run_tests(model_config: dict, test_func, test_cases: dict):
    result = {}
    for stage in ["prefill", "decode"]:
        result[stage] = {}
        for tp_size in test_cases["tp_size"]:

            if test_func == test_router and tp_size > 1:
                continue

            result[stage][tp_size] = {}
            for bs in test_cases["batch_size"]:
                if stage == "prefill":
                    seq_len, max_seq_len = 128, 256
                else:
                    seq_len, max_seq_len = 1, 256

                params = {
                    "tp_size": tp_size,
                    "batch_size": bs,
                    "seq_len": seq_len,
                    "max_seq_len": max_seq_len,
                }

                latency = test_func(model_config, params)
                result[stage][tp_size][bs] = round(latency, 3)

    return result


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model-onfig", type=str)
    # args = parser.parse_args()

    model_config_path = "/mnt/llm_team/merlin_mixtral_weights/v0/config.json"

    with open(model_config_path) as f:
        model_config = json.load(f)

    test_cases = {"tp_size": [1, 2, 4, 6, 8], "batch_size": [1, 2, 4, 8, 16, 32, 64]}

    results = {}
    results["expert_matmul"] = run_tests(model_config, test_expert, test_cases)
    results["router"] = run_tests(model_config, test_router, test_cases)
    results["qkvo"] = run_tests(model_config, test_qkvo, test_cases)
    results["repeat_kv"] = run_tests(model_config, test_repeat_kv, test_cases)
    results["attn_score"] = run_tests(model_config, test_attn_score, test_cases)

    filename = "computation.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
