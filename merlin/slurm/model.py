import argparse
import inspect
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from torch import nn
from xformers.ops.fmha import memory_efficient_attention  # type: ignore
from xformers.ops.fmha.attn_bias import (  # type: ignore
    AttentionBias,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
)

# Environment variables set by torch.distributed.launch
SLURM_PROCID = int(os.environ["SLURM_PROCID"])
GROUP_RANK = int(os.environ["GROUP_RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])


def main():
    print(f"SLURM_PROCID: {SLURM_PROCID}; "
          f"GROUP_RANK: {GROUP_RANK}; "
          f"LOCAL_RANK: {LOCAL_RANK}; "
          f"WORLD_SIZE: {WORLD_SIZE}; "
          f"WORLD_RANK: {WORLD_RANK}\n", flush=True)
    gpu = torch.device(f"cuda:{LOCAL_RANK}")
    dist.init_process_group(
        "nccl", rank=WORLD_RANK, world_size=WORLD_SIZE, device_id=gpu
    )
    group = dist.new_group(list(range(WORLD_SIZE)), use_local_synchronization=True)
    dist.barrier()
    dist.destroy_process_group()
    print("done")


if __name__ == "__main__":
    main()
