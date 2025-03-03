import argparse
import json
import time

import torch

def print_and_save_res(
    title: str, inputs: list[int], avg_latencies: list[float], filename: str
):
    data = {}
    print("-" * 20)
    print(title)
    print("-" * 20)
    print("n_tokens, latency_ms, ")
    for ins, latency in zip(inputs, avg_latencies):
        latency = round(latency, 3)
        data[ins] = latency
        print(f"{ins}, {latency}, ")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

dtype = torch.bfloat16
device = torch.device("cuda:0")
model_d = 4096
interm_d = 14336
n_layers = 32
n_experts = 8
n_warmups, n_samples = 5, 500

def ceildiv(a, b):
    # from: https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return -(a // -b)


def run_workload(tp_size: int, n_tokens: int):
    adjusted_d = ceildiv(interm_d, tp_size)
    
    x = torch.ones((n_tokens, model_d), dtype=dtype, device=device)
    
    # TODO: how many weights do we need to load?
    # initialize weights
    n_ws = n_layers * 3
    ws = [torch.rand(size=(model_d, adjusted_d), dtype=dtype, device=device) for _ in range(n_ws)]

    for _ in range(n_warmups):
        for w in ws:
            y = x @ w
    

    torch.cuda.synchronize(device=device)
    tic = time.time()
    for _ in range(n_samples):
        for w in ws:
            y = x @ w

    torch.cuda.synchronize(device=device)
    latency = (time.time() - tic) * 1000 / (n_samples * n_ws)
    
    return latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-n_tokens", type=int, default=128)
    args = parser.parse_args()
    
    avg_latencies = []
    input_sizes = []
    n_tokens = 1
    while n_tokens <= args.max_n_tokens:
        latency = run_workload(tp_size=2, n_tokens=n_tokens)
        avg_latencies.append(latency)
        input_sizes.append(n_tokens)
        n_tokens = n_tokens*2
    
    print_and_save_res("AVG SINGLE EXPERT MATMUL LATENCY", input_sizes, avg_latencies, "expert_matmul.json")
