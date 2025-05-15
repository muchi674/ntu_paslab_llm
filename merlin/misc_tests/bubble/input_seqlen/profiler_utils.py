import torch

def profile_range(range_name=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            torch.cuda.nvtx.range_push(range_name)
            result = func(*args, **kwargs)
            torch.cuda.nvtx.range_pop()
            return result
        return wrapper
    return decorator