from safetensors.torch import load_file
from safetensors import safe_open
import torch
from pathlib import Path

class MixtralExpertTPPartitioner:
    def __init__(self, safetensor_path: str, output_path: str, tp_size: int = 4):
        self.safetensor_path = safetensor_path
        self.output_path = Path(output_path)
        self.tp_size = tp_size
        self.output_path.mkdir(parents=True, exist_ok=True)

    def run(self):
        tp_partitions = {i: {} for i in range(self.tp_size)}  # accumulate per TP

        with safe_open(self.safetensor_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if ".experts." not in key or not key.endswith(".weight"):
                    continue

                # Parse layer/expert/wX info
                parts = key.split(".")
                li = int(parts[2])
                ep = int(parts[5])
                w_id = parts[6]  # "w1", "w2", etc.

                print(f"Processing {key}")
                tensor = load_file(self.safetensor_path, device="cpu", tensor_names=[key])[key]
                slices = torch.chunk(tensor, self.tp_size, dim=0)

                for tp_id, slice_ in enumerate(slices):
                    name = f"{li}.{ep}.{w_id}"
                    tp_partitions[tp_id][name] = slice_

                del tensor  # free memory

        # Save 4 TP files
        for tp_id, data in tp_partitions.items():
            out_file = self.output_path / f"experts-{tp_id}.pt"
            torch.save(data, out_file)
            print(f"Saved {out_file} with {len(data)} tensors")

# Usage
partitioner = MixtralExpertTPPartitioner(
    safetensor_path="../../mnt/data2/llm_team/Mixtral-8x22B-Instruct-v0.1/",
    output_path="../../mnt/data2/llm_team/merlin_mixtral_8x22B_weight/",
    tp_size=4
)
partitioner.run()


from safetensors.torch import safe_open
from pathlib import Path
import torch

class MixtralShardedPartitioner:
    def __init__(self, index_json_path, output_path, tp_size=4):
        self.index_path = Path(index_json_path)
        self.output_path = Path(output_path)
        self.tp_size = tp_size
        self.output_path.mkdir(parents=True, exist_ok=True)

    def run(self):
        tp_partitions = {i: {} for i in range(self.tp_size)}

        with safe_open(self.index_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if ".experts." not in key or not key.endswith(".weight"):
                    continue

                parts = key.split(".")
                li = int(parts[2])
                ep = int(parts[5])
                w_id = parts[6]  # w1 / w2 / w3

                print(f"Processing {key}")
                tensor = f.get_tensor(key)  # Lazy-loaded from correct shard
                slices = torch.chunk(tensor, self.tp_size, dim=0)

                for tp_id, slice_ in enumerate(slices):
                    name = f"{li}.{ep}.{w_id}"
                    tp_partitions[tp_id][name] = slice_

        # Save 4 files
        for tp_id, data in tp_partitions.items():
            path = self.output_path / f"experts-{tp_id}.pt"
            torch.save(data, path)
            print(f"Saved {path} with {len(data)} items")

# Usage
partitioner = MixtralShardedPartitioner(
    index_json_path="path/to/model.safetensors.index.json",
    output_path="./experts_tp4",
    tp_size=4
)
partitioner.run()
