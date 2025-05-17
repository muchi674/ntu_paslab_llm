import time

import torch

dtype = torch.bfloat16
device = torch.device("cuda:0")
n_warmups, n_samples = 1000, 20000
m = 128 * 4
x1 = torch.rand(size=(m, m), dtype=dtype, device=device)
w1 = torch.rand(size=(m, m), dtype=dtype, device=device)
w2 = torch.rand(size=(m, m), dtype=dtype, device=device)

def perfect(n: int):
    for _ in range(n):
        torch.cuda.nvtx.range_push("cycle")
        y = x1 @ w1
        y = y @ w2
        torch.cuda.nvtx.range_pop()

def control_flow(n: int):
    for _ in range(n):
        torch.cuda.nvtx.range_push("cycle")
        y = x1 @ w1
        # control flow
        if torch.sum(y) == 0:
            continue
        y = y @ w2
        torch.cuda.nvtx.range_pop()

def cpu_operations(n: int):
    for _ in range(n):
        torch.cuda.nvtx.range_push("cycle")
        y = x1 @ w1
        # control flow
        sum(val for val in range(m))
        y = y @ w2
        torch.cuda.nvtx.range_pop()

def main():
    # perfect(n_warmups)
    # control_flow(n_warmups)
    cpu_operations(n_warmups)

    torch.cuda.synchronize(device=device)
    torch.cuda.cudart().cudaProfilerStart()
    tic = time.time()

    # perfect(n_samples)
    # control_flow(n_samples)
    cpu_operations(n_samples)

    torch.cuda.synchronize(device=device)
    total_latency = (time.time() - tic) * 1000
    torch.cuda.cudart().cudaProfilerStop()
    print(f"total time: {total_latency:.2f} ms")
    print(f"per sample time: {(total_latency / n_samples):.2f} ms")

if __name__ == "__main__":
    main()
