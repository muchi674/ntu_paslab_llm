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
        total_token=128,
    )
    model.eval()

    messages = [
        {
            "role": "system",
            "content": "to be, or not to be",
        }
    ]
    input_ids: torch.Tensor = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(gpu)
    output_ids = model.eagenerate(input_ids, temperature=0.5, max_new_tokens=512)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"RESPONSE:")
    print(output)
