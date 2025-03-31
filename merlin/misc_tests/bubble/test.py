import os
import torch

from event_timer import CrossNodeEventTimer

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])

timer = CrossNodeEventTimer(local_rank=LOCAL_RANK, world_size=WORLD_SIZE, world_rank=WORLD_RANK)

NUM_COMP_KERNEL = 10
NUM_COMM_KERNEL = 1

