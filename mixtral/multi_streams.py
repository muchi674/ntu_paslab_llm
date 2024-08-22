import torch
input = []
batch_size = 64
shape = (batch_size, 64, 256)
for i in range (0, 4):
    input.append(batch)
input = [in_.pin_memory() for in_ in input]

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
s3 = torch.cuda.Stream()
s4 = torch.cuda.Stream()

torch.cuda.synchronize()
cpu_device = torch.device('cpu')
device = torch.device('cuda:0')

with torch.cuda.stream(s1):
    curr_batch = input[0].to(device,non_blocking = True)
    curr_batch = torch.exp (curr_batch)
    curr_batch = torch.exp (curr_batch)
    curr_batch = torch.exp (curr_batch)
    input[0].copy_(curr_batch,non_blocking = True)

with torch.cuda.stream(s2):
    curr_batch = input[1].to(device,non_blocking = True)
    curr_batch = torch.exp (curr_batch)
    curr_batch = torch.exp (curr_batch)
    curr_batch = torch.exp (curr_batch)
    input[1].copy_(curr_batch,non_blocking = True)


with torch.cuda.stream(s3):
    curr_batch = input[2].to(device,non_blocking = True)
    curr_batch = torch.exp (curr_batch)
    curr_batch = torch.exp (curr_batch)
    curr_batch = torch.exp (curr_batch)
    input[2].copy_(curr_batch,non_blocking = True)


with torch.cuda.stream(s4):
    curr_batch = input[3].to(device,non_blocking = True)
    curr_batch = torch.exp (curr_batch)
    curr_batch = torch.exp (curr_batch)
    curr_batch = torch.exp (curr_batch)
    input[3].copy_(curr_batch,non_blocking = True)


torch.cuda.synchronize()
