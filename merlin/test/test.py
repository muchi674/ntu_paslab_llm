import time
import nvtx
import torch
import torch.nn.functional as F
import argparse

def test(input_size):
    device = torch.device("cuda:0")
    inputs = torch.ones((2, 4096), dtype=torch.bfloat16, device=device)
    gate_logits = torch.rand((input_size, 8), dtype=torch.bfloat16, device=device)
    #print(gate_logits)
    weights, selected_experts = torch.topk(gate_logits, 2)
    #print(weights)
    #print(selected_experts)
    weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)

    # # warmup
    # for _ in range(100):
    #     for ei in range(8):
    #         batch_idx, nth_expert = torch.where(selected_experts == ei)
    #         if torch.numel(batch_idx) == 0:
    #             continue

    # torch.cuda.synchronize()
    # torch.cuda.cudart().cudaProfilerStart()
    # torch.cuda.nvtx.range_push("test")
    # tic = time.time()

    # for ei in range(8):
    #     batch_idx, nth_expert = torch.where(selected_experts == ei)
    #     if torch.numel(batch_idx) == 0:
    #         continue

    # torch.cuda.synchronize()
    # torch.cuda.nvtx.range_pop()
    # print(time.time() - tic)
    # torch.cuda.cudart().cudaProfilerStop()

    # warmup
    for _ in range(100):
        tmp = selected_experts.to("cpu")
        bis, nes = [], []
        for ei in range(8):
            batch_idx, nth_expert = torch.where(tmp == ei)
            if torch.numel(batch_idx) > 0:
                bis.append(batch_idx.to(device=device))
                nes.append(nth_expert.to(device=device))

    # torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("test")
    tic = time.time()

    tmp = selected_experts.to("cpu")
    eis, bis, nes = [], [], []
    ### original
    # with nvtx.annotate("data transferr", color="purple"):
    #     for ei in range(8):
    #         batch_idx, nth_expert = torch.where(tmp == ei)
    #         with nvtx.annotate("one sec", color="blue"):
    #             if torch.numel(batch_idx) > 0:
    #                 eis.append(ei)
    #                 with nvtx.annotate("each trans", color="blue"):
    #                     bis.append(batch_idx.to(device=device))
    #                 with nvtx.annotate("each trans", color="blue"):
    #                     nes.append(nth_expert.to(device=device))

    ### one memcpy
    with nvtx.annotate("full range", color="purple"):
        #select_shape = []   # for one memcpy version, save the tensor's size
        
        for ei in range(8):
            batch_idx, nth_expert = torch.where(tmp == ei)
            if torch.numel(batch_idx) > 0:
                eis.append(ei)
                bis.append(batch_idx)
                nes.append(nth_expert)
                #select_shape.append(len(batch_idx))
        concat_bis = torch.cat(bis, dim=0)
        concat_nes = torch.cat(bis, dim=0)
        with nvtx.annotate("data transferr", color="blue"):
            
            concat_bis_cpu = concat_bis.to(device=device)
            #bis = torch.split(concat_bis_cpu, [len(t) for t in bis])

            concat_nes_cpu = concat_nes.to(device=device)
            #nes = torch.split(concat_nes_cpu, [len(t) for t in nes])

    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    print((time.time() - tic) * 1000)
    torch.cuda.cudart().cudaProfilerStop()


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Run the program with different input sizes.')
    parser.add_argument('--input_size', type=int, required=True, help='Input size to use')
    
    args = parser.parse_args()
    
    # Run the program with the specified input size
    test(args.input_size)

if __name__ == "__main__":
    main()