import time
from pathlib import Path

import torch

from transformers import AutoTokenizer
from EAGLE.eagle.model.ea_model import EaModel

if __name__ == "__main__":
    draft_model_path = Path("/home/joe/EAGLE-LLaMA3-Instruct-8B")
    target_model_path = Path("/home/joe/Meta-Llama-3-8B-Instruct")
    gpu = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    model = EaModel.from_pretrained(
        base_model_path=target_model_path,
        ea_model_path=draft_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=gpu,
        total_token=-1,
    )
    model.eval()

    messages = [
        {
            "role": "system",
            "content": "Imagine you are an experienced Ethereum developer tasked with creating a smart contract for a blockchain messenger. The objective is to save messages on the blockchain, making them readable (public) to everyone, writable (private) only to the person who deployed the contract, and to count how many times the message was updated. Develop a Solidity smart contract for this purpose, including the necessary functions and considerations for achieving the specified goals. Please provide the code and any relevant explanations to ensure a clear understanding of the implementation.",
        }
    ]
    input_ids: torch.Tensor = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(gpu)

    # warmup
    for _ in range(5):
        model.eagenerate(input_ids, temperature=0.5, max_new_tokens=512, is_llama3=True)

    torch.cuda.synchronize()
    start_time = time.time()

    output_ids = model.eagenerate(input_ids, temperature=0.5, max_new_tokens=512, is_llama3=True, profile=True)

    torch.cuda.synchronize()
    total_time = time.time() - start_time

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"TOTAL TIME: {total_time:.2f}")
    print(f"RESPONSE:")
    print(output)
