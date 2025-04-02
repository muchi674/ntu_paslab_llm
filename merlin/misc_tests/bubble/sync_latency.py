import os
import torch
import torch.distributed as dist
from torch import nn

from event_timer import CrossNodeEventTimer

# Environment variables set by torch.distributed.launch
SLURM_PROCID = int(os.environ["SLURM_PROCID"])
GROUP_RANK = int(os.environ["GROUP_RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])

timer = CrossNodeEventTimer(local_rank=LOCAL_RANK, world_size=WORLD_SIZE, world_rank=WORLD_RANK)

NUM_COMP_KERNEL = 10
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
        self.kernel_type = 'comp'
        self.net = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, input):
        return self.net(input)

class AllReduceKernel(nn.Module):
    def __init__(self):
        self.kernel_type = 'comm'

    def forward(self, input):
        timer.record_end()
        dist.all_reduce(input, op=dist.ReduceOp.SUM)
        dist.barrier()

        timer.acc_elapsed_time()

        return input

class WorkLoadBase(nn.Module):
    def __init__(self, device: torch.device):
        self.device = device

        self.hidden_size = 4096
        self.intermediate_size = 14336

        # self.max_position_embeddings = 32768 # max context length, use 128

        self.kernels = self._gen_mock_net()
        
    def forward(self, input):
        for kernel in self.kernels:
            if kernel.kernel_type == 'comp':
                input = kernel(input)
            elif kernel.kernel_type == 'comm':
                input = kernel(input)

    def gen_mock_input(self, batch_size: int) -> torch.Tensor:
        seqlen = 1 # prefill 128, decode 1
        input_dim = [batch_size, seqlen, self.hidden_size]
        return torch.rand(input_dim, dtype=torch.bfloat16)
    
    def _gen_mock_net(self) -> nn.ModuleList:
        kernels = nn.ModuleList()
        
        kernels.append(CompKernel(self.hidden_size, self.intermediate_size))
        for i in range(NUM_COMP_KERNEL-2):
            kernels.append(CompKernel(self.intermediate_size, self.intermediate_size))
        kernels.append(CompKernel(self.intermediate_size, self.hidden_size))

        kernels.append(AllReduceKernel())       

        return kernels
    

class SyncLatencyMicroBenchmark:
    def __init__(self, workload: WorkLoadBase):
        self.workload = workload

    @torch.inference_mode()
    def run(self, num_batches:int = 32, batch_size: int = 1, max_tokens: int = 40):
        self.workload = self.workload.eval()

        timer.record_start(self.device)

        input = self.workload.gen_mock_input(batch_size=batch_size)
        for i in range(num_batches):
            for i in range(max_tokens):
                input = self.workload(input)

                timer.record_elapsed_time()

            timer.flush_buffer(isPrefill=False)

        timer.all_gather(num_batches, max_tokens, None)
        
    @property
    def device(self) -> torch.device:
        return self.workload.device

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
    # group = dist.new_group(list(range(WORLD_SIZE)), use_local_synchronization=True)

    torch.cuda.cudart().cudaProfilerStart()
    timer.reset()
    # =============================================================================
    # TODO
    
    workload = WorkLoadBase(device=gpu)
    mb = SyncLatencyMicroBenchmark(workload=workload)
    mb.run()
    timer.get_sync_latency()
    
    # =============================================================================
    torch.cuda.cudart().cudaProfilerStop()

    dist.barrier()
    dist.destroy_process_group()
    print("done")


if __name__ == "__main__":
    main()
