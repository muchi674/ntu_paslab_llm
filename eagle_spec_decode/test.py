import gc
import time
from pathlib import Path

import torch

from transformers import AutoTokenizer
from EAGLE.eagle.model.ea_model import EaModel as OnChipEaModel
from gpu_only_offloading.eagle.model.ea_model import EaModel as OffloadedEaModel
from static_collab.eagle.model.ea_model import EaModel as StaticCollabEaModel

def on_chip_test(gpu: torch.device):
    draft_model_path = Path("/home/joe/EAGLE-LLaMA3-Instruct-8B")
    target_model_path = Path("/home/joe/Meta-Llama-3-8B-Instruct")

    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    model = OnChipEaModel.from_pretrained(
        base_model_path=target_model_path,
        ea_model_path=draft_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=gpu,  # same as default
        total_token=59
    )

    return tokenizer, model

def gpu_only_offloading_test(cpu: torch.device, gpu: torch.device):
    draft_model_path = Path("/home/joe/EAGLE-LLaMA3-Instruct-70B")
    target_model_path = Path("/home/joe/Meta-Llama-3-70B-Instruct")
    torch.set_default_device(gpu)

    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    model = OffloadedEaModel.from_pretrained(
        base_model_path=target_model_path,
        ea_model_path=draft_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=cpu,
        total_token=59,  # same as default
        draft_device=gpu
    )

    return tokenizer, model

def static_collab_test(cpu: torch.device, gpu: torch.device):
    draft_model_path = Path("/home/joe/EAGLE-LLaMA3-Instruct-70B")
    target_model_path = Path("/home/joe/Meta-Llama-3-70B-Instruct")
    torch.set_default_device(gpu)

    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    model = StaticCollabEaModel.from_pretrained(
        base_model_path=target_model_path,
        ea_model_path=draft_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=cpu,
        total_token=59,  # same as default
        draft_device=gpu
    )

    return tokenizer, model

if __name__ == "__main__":
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")
    # tokenizer, model = on_chip_test(gpu)
    # tokenizer, model = gpu_only_offloading_test(cpu, gpu)
    tokenizer, model = static_collab_test(cpu, gpu)
    model.eval()
    print("FINISHED INITIALIZING MODEL")

    messages = [
        {
            "role": "system",
            "content": "Imagine you are an experienced Ethereum developer tasked with creating a smart contract for a blockchain messenger. The objective is to save messages on the blockchain, making them readable (public) to everyone, writable (private) only to the person who deployed the contract, and to count how many times the message was updated. Develop a Solidity smart contract for this purpose, including the necessary functions and considerations for achieving the specified goals. Please provide the code and any relevant explanations to ensure a clear understanding of the implementation.",
        }
    ]
    input_ids: torch.Tensor = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(gpu)

    # torch.compile(model)

    # warmup
    for _ in range(3):
        model.eagenerate(input_ids, temperature=0.5, max_new_tokens=1, is_llama3=True)

    start_time = time.time()

    output_ids = model.eagenerate(input_ids, temperature=0.5, max_new_tokens=128, is_llama3=True, profile=True)

    torch.cuda.synchronize()
    total_time = time.time() - start_time

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"TOTAL TIME: {total_time:.2f}")
    print(f"RESPONSE:")
    print(output)
