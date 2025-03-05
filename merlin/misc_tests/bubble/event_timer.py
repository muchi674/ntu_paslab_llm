import torch
import torch.distributed as dist

class CrossNodeEventTimer:
    def __init__(self, local_rank, world_size, world_rank):
        self.local_rank = local_rank
        self.world_size = world_size
        self.world_rank = world_rank
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

        self.records = {f"{world_rank}_p": [], f"{world_rank}_d": []}
        self.buff = []
        self.temp = 0

        self.out_p = None
        self.out_d = None

        self.device = torch.device(f"cuda:{local_rank}")

    def record_start(self, device: torch.device):
        self.start_event.record(torch.cuda.current_stream(device))
    
    def record_end(self, device: torch.device):
        self.end_event.record(torch.cuda.current_stream(device))
        torch.cuda.synchronize(device)

    def acc_elapsed_time(self) -> float:
        '''return accumulated elapsed time'''
        duration = self.start_event.elapsed_time(self.end_event)
        self.temp += duration

        return self.temp
    
    def record_elapsed_time(self):
        '''append accumulated elapsed time to buffer'''
        self.buff.append(self.temp)
        self.temp = 0

        return self.buff
    
    def flush_buffer(self, isPrefill=False):
        '''append buffer to records'''
        self.records[f"{self.world_rank}_p" if isPrefill else f"{self.world_rank}_d"].append(self.buff)
        self.buff = []

    def reset(self):
        self.buff = []
        self.temp = 0
        self.records = {f"{self.world_rank}_p": [], f"{self.world_rank}_d": []}

    def all_gather(self, num_batches, max_tokens, group=None):
        '''
        gather from all devices to obtain duration of prefill(all tokens) and decode(token-wise) computation\n
        return with shape of (#devices, #batches) for prefill (out_p)\n
        return with shape of (#devices, #batches, #tokens) for decode (out_d)
        '''
        print("prefill",torch.tensor(self.records[f"{self.world_rank}_p"]).shape)
        print("decode",torch.tensor(self.records[f"{self.world_rank}_d"]).shape)

        self.out_p = torch.zeros((self.world_size, num_batches), device=self.device)
        self.out_d = torch.zeros((self.world_size, num_batches, max_tokens), device=self.device)

        dist.all_gather_into_tensor(self.out_p, torch.tensor(self.records[f"{self.world_rank}_p"], device=self.device), group)
        dist.all_gather_into_tensor(self.out_d, torch.tensor(self.records[f"{self.world_rank}_d"], device=self.device), group)

        return self.out_p, self.out_d

    def get_sync_latency(self):
        """
        Example usage of torch.amax and torch.amin for element-wise comparison.

        Example:
            >>> import torch
            >>> d = [
            ...     [
            ...         [5, 10],
            ...         [3, 15]
            ...     ],
            ...     [
            ...         [2, 4],
            ...         [6, 8]
            ...     ]
            ... ]
            >>> tensor_d = torch.tensor(d)

        Compute element-wise max along dim=0
            >>> torch.amax(tensor_d, dim=0)
            tensor([
                [5, 10],  # Max of [5,2] and [10,4]
                [6, 15]   # Max of [3,6] and [15,8]
            ])

        Compute element-wise min along dim=0
            >>> torch.amin(tensor_d, dim=0)
            tensor([
                [2, 4],  # Min of [5,2] and [10,4]
                [3, 8]   # Min of [3,6] and [15,8]
            ])
        """

        max_p = torch.amax(self.out_p, dim=0)
        min_p = torch.amax(self.out_p, dim=0)
        self.out_p = max_p - min_p

        max_d = torch.amax(self.out_d, dim=0)
        min_d = torch.amin(self.out_d, dim=0)
        self.out_d = max_d - min_d

        # print(self.out_p, self.out_d, self.out_d.dtype)
        print("average bubble-caused sync latency (ms) per batch")
        print(self.out_d.mean(dim=1, keepdim=True))
        print(f"average bubble-caused sync latency (ms) for #batches = {self.out_d.shape[0]}")
        print(self.out_d.mean(dim=1).mean())