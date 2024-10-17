import torch

if __name__ == "__main__":
    gpu_0 = torch.device("cuda:0")
    gpu_1 = torch.device("cuda:1")
    arr = torch.ones(2, 8192, 32768).pin_memory()

    for _ in range(5):
        arr.to(gpu_0)

    for _ in range(5):
        arr.to(gpu_1)
