import torch
import numpy
v = torch.randint(20, 50 ,(1, 8))

c = v[0].numpy()
for i in range(1, c.shape[0]):
    print(i)
    
print(c)