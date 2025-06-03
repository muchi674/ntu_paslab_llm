import os
import torch
import torch.distributed as dist
from torch import nn
from argparse import ArgumentParser

from event_timer import CrossNodeEventTimer

# Environment variables set by torch.distributed.launch
SLURM_PROCID = int(os.environ["SLURM_PROCID"])
GROUP_RANK = int(os.environ["GROUP_RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])

timer = CrossNodeEventTimer(local_rank=LOCAL_RANK, world_size=WORLD_SIZE, world_rank=WORLD_RANK)

NUM_COMP_KERNEL = 10
TIMES_COPIES = 4 # 2^16
# NUM_COMM_KERNEL = 1

'''
{
  "architectures": [
    "MixtralForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 32768,
  "model_type": "mixtral",
  "num_attention_heads": 32,
  "num_experts_per_tok": 2,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "num_local_experts": 8,
  "output_router_logits": false,
  "rms_norm_eps": 1e-05,
  "rope_theta": 1000000.0,
  "router_aux_loss_coef": 0.02,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.36.0.dev0",
  "use_cache": true,
  "vocab_size": 32000
}
'''

class CompKernel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.kernel_type = 'comp'
        self.net = nn.Linear(input_dim, output_dim, bias=False, dtype=torch.bfloat16)

    def forward(self, input):
        return self.net(input)

class AllReduceKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_type = 'comm'

    def forward(self, input):
        dist.all_reduce(input, op=dist.ReduceOp.SUM)
        dist.barrier()

        return input

class WorkLoadBase(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

        self.hidden_size = 4096
        self.intermediate_size = 14336

        # self.max_position_embeddings = 32768 # max context length, use 128

        self.kernels = self._gen_mock_net()
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        for i, kernel in enumerate(self.kernels):
            if kernel.kernel_type == 'comp':
                input = kernel(input)
            elif kernel.kernel_type == 'comm':
                timer.record_end()

                input = kernel(input)

                timer.acc_elapsed_time()
                if i != len(self.kernels) - 1:
                    timer.record_start()

        return input

    def gen_mock_input(self, batch_size: int, seqlen: int = 1) -> torch.Tensor:
        # for seqlen: prefill 128, decode 1
        input_dim = [batch_size, seqlen, self.hidden_size]
        return torch.rand(input_dim, dtype=torch.bfloat16, device=self.device)
    
    def _gen_mock_net(self) -> nn.ModuleList:
        kernels = nn.ModuleList()
        
        kernels.append(CompKernel(self.hidden_size, self.intermediate_size))
        for i in range(NUM_COMP_KERNEL-2):
            kernels.append(CompKernel(self.intermediate_size, self.intermediate_size))
        kernels.append(CompKernel(self.intermediate_size, self.hidden_size))

        kernels.append(AllReduceKernel())       

        import copy
        for i in range(TIMES_COPIES):
            kernels.extend(copy.deepcopy(kernels))
        return kernels
    

class SyncLatencyMicroBenchmark:
    def __init__(self, workload: WorkLoadBase):
        self.workload = workload

    @torch.inference_mode()
    def run_prefill(self, num_batches:int = 32, batch_size: int = 1):
        self.workload = self.workload.eval()

        input = self.workload.gen_mock_input(batch_size=batch_size, seqlen=128)
        for i in range(num_batches):
            timer.record_start(self.device)
            input = self.workload(input)

            timer.record_elapsed_time()
            timer.flush_buffer(isPrefill=True)

    @torch.inference_mode()
    def run_decode(self, num_batches:int = 32, batch_size: int = 1, max_tokens: int = 128):
        self.workload = self.workload.eval()

        
        input = self.workload.gen_mock_input(batch_size=batch_size, seqlen=1)
        for i in range(num_batches):
            for i in range(max_tokens):
                timer.record_start(self.device)
                # input = self.workload(input)
                output: torch.Tensor = self.workload(input)
                next_token = output[:, -1, :]
                input = torch.cat([input, next_token], dim=1)

                timer.record_elapsed_time()

            timer.flush_buffer(isPrefill=False)

    @torch.inference_mode()
    def dry_run_decode(self, num_batches:int = 32, batch_size: int = 1, max_tokens: int = 40):
        self.workload = self.workload.eval()

        input = self.workload.gen_mock_input(batch_size=batch_size, seqlen=1)
        for i in range(num_batches):
            for i in range(max_tokens):
                output: torch.Tensor = self.workload(input)
                next_token = output[:, -1, :]
                input = torch.cat([input, next_token], dim=1)
        
    @property
    def device(self) -> torch.device:
        return self.workload.device

def main(
        num_batches: int =32,
        batch_size: int = 1,
        max_tokens: int = 40
    ):
    print(f"SLURM_PROCID: {SLURM_PROCID}; "
          f"GROUP_RANK: {GROUP_RANK}; "
          f"LOCAL_RANK: {LOCAL_RANK}; "
          f"WORLD_SIZE: {WORLD_SIZE}; "
          f"WORLD_RANK: {WORLD_RANK}\n", flush=True)
    gpu = torch.device(f"cuda:{LOCAL_RANK}")
    dist.init_process_group(
        "nccl", rank=WORLD_RANK, world_size=WORLD_SIZE, device_id=gpu
    )
    # group = dist.new_group(list(range(WORLD_SIZE)), use_local_synchronization=True)
    workload = WorkLoadBase(device=gpu).to(device=gpu)
    mb = SyncLatencyMicroBenchmark(workload=workload)
    # mb.run_decode(num_batches=16, batch_size=batch_size, max_tokens=max_tokens) # warmup
    mb.dry_run_decode(num_batches=16, batch_size=batch_size, max_tokens=max_tokens)
    # timer.reset()

    torch.cuda.cudart().cudaProfilerStart()
    # =============================================================================
    # TODO
    # mb.run_prefill(num_batches=num_batches, batch_size=batch_size)
    # mb.run_decode(num_batches=num_batches, batch_size=batch_size, max_tokens=max_tokens)
    mb.dry_run_decode(num_batches=num_batches, batch_size=batch_size, max_tokens=max_tokens)

    # timer.all_gather(num_batches, max_tokens, None)

    if WORLD_RANK == 0:
        # timer.get_sync_latency()
        pass
    
    # =============================================================================
    torch.cuda.cudart().cudaProfilerStop()

    dist.barrier()
    dist.destroy_process_group()
    print("done")


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--model-path", type=str)
    # parser.add_argument("--node-id", type=int)
    # parser.add_argument("--prompt", type=str)
    # parser.add_argument("--prompt-path", type=str)
    # parser.add_argument("--n-prompts", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    # parser.add_argument("--hide-resp", action="store_true")
    args = parser.parse_args()

    main(batch_size=args.batch_size, max_tokens=args.max_tokens)
