import torch
import time

device = "cuda:0"
hidden_dim = 4096

# intermediate_dims = [2**i for i in range(15)]
# num_tokens = [2**i for i in range(17)]

intermediate_dims = [7168]
num_tokens = [256]


num_iter = 400
for dim in intermediate_dims:
    print(f"intermediate_dim = {dim}:")
    for n in num_tokens:
        # warm up
        for _ in range(100):
            x = torch.rand((n, hidden_dim), dtype=torch.bfloat16, device=device)
            e = torch.rand((hidden_dim, dim), dtype=torch.bfloat16, device=device)
            y = x @ e
        
        total_time = 0
        for _ in range(num_iter):
            x = torch.rand((n, hidden_dim), dtype=torch.bfloat16, device=device)
            w1 = torch.rand((hidden_dim, dim), dtype=torch.bfloat16, device=device)
            w2 = torch.rand((dim, hidden_dim), dtype=torch.bfloat16, device=device)
            w3 = torch.rand((hidden_dim, dim), dtype=torch.bfloat16, device=device)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            # y = ((x @ w1) * (x @ w3)) @ w2
            y = x @ w1
            end.record()

            torch.cuda.synchronize()
            total_time += start.elapsed_time(end)

        print(f"{total_time/num_iter:.2f}", end=" ")
    print("\n")

