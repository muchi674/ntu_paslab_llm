import torch

device = torch.device("cuda:0")
dtype = torch.bfloat16

# s1 = torch.cuda.default_stream(device=device)
# s2 = torch.cuda.Stream()
# event = torch.cuda.Event()
# x = torch.ones((2, 2), dtype=dtype, device=device)

# x.add_(1)
# event.record()

# with torch.cuda.stream(s2):
#     event.wait()
#     x.div_(2)

# torch.cuda.synchronize(device=device)
# print(x)

# **************************************************

# N = 4
# qs = [torch.rand((4096, 4096), dtype=dtype, device=device) for _ in range(N)]
# os = [torch.rand((4096, 4096), dtype=dtype, device=device) for _ in range(N)]
# ups = [torch.rand((14336, 4096), dtype=dtype, device=device) for _ in range(N)]
# down = [torch.rand((4096, 14336), dtype=dtype, device=device) for _ in range(N)]
# xs = [torch.ones((4, 4096), dtype=dtype, device=device) for _ in range(N)]

# attn_stream = torch.cuda.default_stream(device=device)
# experts_stream = torch.cuda.Stream()
# attn_events = [torch.cuda.Event() for _ in range(N)]
# experts_events = [torch.cuda.Event() for _ in range(N)]

# def first_attn():
#     with torch.cuda.stream(attn_stream):
#         (xs[0] @ qs[0].T) @ os[0].T
#         attn_events[0].record()

# def subseq_attn(i: int):
#     with torch.cuda.stream(attn_stream):
#         experts_events[i - 1].wait()
#         (xs[i] @ qs[i].T) @ os[i].T
#         attn_events[i].record()

# def experts(i: int):
#     with torch.cuda.stream(experts_stream):
#         attn_events[i].wait()
#         (xs[i] @ ups[i].T) @ down[i].T
#         experts_events[i].record()

# torch.cuda.synchronize(device=device)
# torch.cuda.cudart().cudaProfilerStart()

# first_attn()
# experts(0)
# for i in range(1, N):
#     subseq_attn(i)
#     experts(i)

# torch.cuda.synchronize(device=device)
# torch.cuda.cudart().cudaProfilerStop()

# **************************************************

N = 10
qs = [torch.rand((4096, 4096), dtype=dtype, device=device) for _ in range(N)]
os = [torch.rand((4096, 4096), dtype=dtype, device=device) for _ in range(N)]
ups = [torch.rand((14336, 4096), dtype=dtype, device=device) for _ in range(N)]
down = [torch.rand((4096, 14336), dtype=dtype, device=device) for _ in range(N)]
xs = [torch.ones((4, 4096), dtype=dtype, device=device) for _ in range(N)]

attn_stream = torch.cuda.Stream()
experts_stream = torch.cuda.Stream()
attn_events = [torch.cuda.Event() for _ in range(N)]
experts_events = [torch.cuda.Event() for _ in range(N)]

def init_cuda():
    for i in range(N):
        (xs[i] @ qs[i].T) @ os[i].T
        (xs[i] @ ups[i].T) @ down[i].T

init_cuda()

def first_attn():
    with torch.cuda.stream(attn_stream):
        (xs[0] @ qs[0].T) @ os[0].T
        attn_events[0].record()

def subseq_attn(i: int):
    (xs[i] @ qs[i].T) @ os[i].T

def experts(i: int):
    with torch.cuda.stream(experts_stream):
        attn_events[i].wait()
        (xs[i] @ ups[i].T) @ down[i].T
        experts_events[i].record()

def draw_graphs():
    graphs = []
    for i in range(1, N):
        graphs.append(torch.cuda.CUDAGraph())
        with torch.cuda.graph(graphs[-1], stream=attn_stream):
            subseq_attn(i)
    return graphs

def forward(graphs: list[torch.cuda.CUDAGraph]):
    first_attn()
    experts(0)
    for i in range(1, N):
        with torch.cuda.stream(attn_stream):
            experts_events[i - 1].wait()
            graphs[i - 1].replay()
            attn_events[i].record()
        experts(i)

with torch.cuda.device(device=device):
    graphs = draw_graphs()
torch.cuda.synchronize(device=device)
torch.cuda.cudart().cudaProfilerStart()

forward(graphs)

torch.cuda.synchronize(device=device)
torch.cuda.cudart().cudaProfilerStop()
