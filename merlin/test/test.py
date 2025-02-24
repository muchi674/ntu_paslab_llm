import torch

t = torch.randn(3, 4, device='cuda') 

# indices = torch.arange(0, t.size().numel()) 
sorted_values, sorted_indices = t.sort(dim=-1)

print(t)
print(sorted_values)
print(sorted_indices)