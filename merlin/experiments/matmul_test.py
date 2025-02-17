import torch
import time

torch.manual_seed(1234)
device = "cuda:0"
hidden_dim = 4096
num_experts = 8
#intermediate_dims = [2**i for i in range(15)]
num_tokens = [2**i for i in range(15)]
intermediate_dims = [1792, 2389, 3584, 7168, 14336]
# num_tokens = [i for i in range(20, 40)]
# intermediate_dims = [7168]

num_iter = 400
num_warmup_step = 100
for dim in intermediate_dims:
    print(f"intermediate_dim = {dim}:")
    # initialize expert weights
    expert_weights = {}
    for ei in range(num_experts):
        expert_weights[f"{ei}.w1"] = torch.rand(
            dim,
            hidden_dim,
            dtype = torch.bfloat16,
            device=device
        )
        expert_weights[f"{ei}.w2"] = torch.rand(
            dim,
            hidden_dim,
            dtype = torch.bfloat16,
            device=device
        )
        expert_weights[f"{ei}.w3"] = torch.rand(
            dim,
            hidden_dim,
            dtype = torch.bfloat16,
            device=device
        )

    for n in num_tokens:
        x = torch.rand((n, hidden_dim), dtype=torch.bfloat16, device=device)
        for nth_iter in range(num_warmup_step):
            ei = nth_iter % num_experts
            w1 = expert_weights[f"{ei}.w1"].T
            w2 = expert_weights[f"{ei}.w2"]
            w3 = expert_weights[f"{ei}.w3"].T

            y = (torch.nn.functional.silu(x @ w1) * (x @ w3)) @ w2
            # y = x @ w1
        
        torch.cuda.synchronize(device=x.device)
        tic = time.time()
        for nth_iter in range(num_iter):
            ei = nth_iter % num_experts
            w1 = expert_weights[f"{ei}.w1"].T
            w2 = expert_weights[f"{ei}.w2"]
            w3 = expert_weights[f"{ei}.w3"].T

            y = (torch.nn.functional.silu(x @ w1) * (x @ w3)) @ w2
            # y = x @ w1
        
        torch.cuda.synchronize(device=x.device)
        total_time = time.time()-tic
        print(f"{total_time*1000/num_iter:.3f}", end=" ")
    print("\n")

