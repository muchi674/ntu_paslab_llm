import torch
import math

torch.manual_seed(42)

# --- Hyperparameters ---
B, T, D = 32, 256, 16384     # batch, seq, dim
H = 8                   # heads
D_H = D // H            # head dim
FFN_HID = 4096          # hidden dim for expert

device = "cuda"
assert torch.cuda.is_available()

# --- Input ---
x = torch.randn(B, T, D, device=device)

# --- Expert FFN weights ---
W1 = torch.randn(D, FFN_HID, device=device) / math.sqrt(D)
W2 = torch.randn(FFN_HID, D, device=device) / math.sqrt(FFN_HID)

# --- Attention weights ---
Wq = torch.nn.Linear(D, D, bias=False, device=device)
Wk = torch.nn.Linear(D, D, bias=False, device=device)
Wv = torch.nn.Linear(D, D, bias=False, device=device)

# === Streams and event ===
stream_expert = torch.cuda.Stream()
stream_graph = torch.cuda.Stream()
event = torch.cuda.Event()


with torch.cuda.stream(stream_expert):
    x_expert = torch.nn.functional.gelu(x @ W1)
    x_expert = x_expert @ W2
    event.record()  # mark when expert is done

q_in = x_expert.clone()
k_in = x_expert.clone()
v_in = x_expert.clone()
attn_out = torch.empty_like(x)

graph = torch.cuda.CUDAGraph()

def attention_static(q, k, v):
    B, T, D = q.shape
    q = q.view(B, T, H, D_H).transpose(1, 2)  # [B, H, T, D_H]
    k = k.view(B, T, H, D_H).transpose(1, 2)
    v = v.view(B, T, H, D_H).transpose(1, 2)
    attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D_H)
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)
    out = out.transpose(1, 2).reshape(B, T, D)
    return out

torch.cuda.synchronize()
with torch.cuda.graph(graph, stream=stream_graph):
    q = Wq(q_in)
    k = Wk(k_in)
    v = Wv(v_in)
    attn_out.copy_(attention_static(q, k, v))

with torch.cuda.stream(stream_graph):
    stream_graph.wait_event(event)
    q_in.copy_(x_expert)
    k_in.copy_(x_expert)
    v_in.copy_(x_expert)
    graph.replay()

torch.cuda.synchronize()
print("Attn output:", attn_out[0, 0, :10])  # print first token vector
