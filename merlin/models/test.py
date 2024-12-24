import torch

is_finished = torch.tensor([False for _ in range(10)], device=torch.device("cuda:0"))
next_token = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], device=torch.device("cuda:0"))

for _ in range(100):
    is_finished = is_finished | (next_token == 1)

    if is_finished.all():
        break
