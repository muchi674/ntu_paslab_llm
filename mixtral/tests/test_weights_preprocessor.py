from pathlib import Path

import torch
import safetensors.torch

# cpu_outs = safetensors.torch.load_file(
#     "/home/joe/ntu_paslab_llm/mixtral/v0_poc/cpu_outs.safetensors", device="cpu"
# )
# gpu_outs = safetensors.torch.load_file(
#     "/home/joe/ntu_paslab_llm/mixtral/v0_poc/gpu_outs.safetensors", device="cpu"
# )
# for li in range(4):
#     for ei in range(8):
#         cout = cpu_outs[f"{li}.{ei}"]
#         gout = gpu_outs[f"{li}.{ei}"]
#         print(f"{li}.{ei} equality: {torch.equal(cout, gout)}")
#         print(f"cpu_out: {cout}")
#         print(f"gpu_out: {gout}")
#     cout = cpu_outs[f"{li}.h"]
#     gout = gpu_outs[f"{li}.h"]
#     print(f"{li}.h equality: {torch.equal(cout, gout)}")
#     print(f"cpu_out: {cout}")
#     print(f"gpu_out: {gout}")

# folder = Path("/home/joe/Mixtral-8x7B-Instruct-v0.1")
# expected = {}
# for i in range(1, 20):
#     fn = f"model-000{0 if i < 10 else ''}{i}-of-00019.safetensors"
#     expected.update(safetensors.torch.load_file(folder / fn, device="cpu"))
# test = safetensors.torch.load_file(
#     "/home/joe/Mixtral-8x7B-Instruct-v0.1/offloaded/experts.safetensors", device="cpu"
# )
# for li in range(32):
#     for ei in range(8):
#         w1 = expected[f"model.layers.{li}.block_sparse_moe.experts.{ei}.w1.weight"]
#         w2 = expected[f"model.layers.{li}.block_sparse_moe.experts.{ei}.w2.weight"]
#         w3 = expected[f"model.layers.{li}.block_sparse_moe.experts.{ei}.w3.weight"]
#         eq = torch.equal(torch.stack((w1, w2.T, w3), dim=0), test[f"{li}.{ei}"])
#         if not eq:
#             print(f"{li} {ei} equality check failed")
#     print(f"{li} is aight")

expected = torch.load(
    str("/home/joe/Mixtral-8x7B-Instruct-v0.1-Official/consolidated.00.pth"), mmap=True
)
# test = torch.load(
#     str("/home/joe/Mixtral-8x7B-Instruct-v0.1-Official/non-experts.pt"), mmap=True
# )
# for li in range(32):
#     for pi in ["q", "k", "v", "o"]:
#         k = f"layers.{li}.attention.w{pi}.weight"
#         if not torch.equal(expected[k], test[k]):
#             print(f"inequality detected in li: {li}, pi: {pi}")

test = torch.load(
    str("/home/joe/Mixtral-8x7B-Instruct-v0.1-Official/experts.pt"), mmap=True
)
for li in range(32):
    for ei in range(8):
        w1 = expected[f"layers.{li}.feed_forward.experts.{ei}.w1.weight"]
        w2 = expected[f"layers.{li}.feed_forward.experts.{ei}.w2.weight"]
        w3 = expected[f"layers.{li}.feed_forward.experts.{ei}.w3.weight"]
        if not torch.equal(torch.stack((w1, w2.T, w3), dim=0), test[f"{li}.{ei}"]):
            print(f"inequality detected in li: {li}, ei: {ei}")
