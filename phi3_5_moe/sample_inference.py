import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    "/home/joe/Phi-3.5-MoE-instruct",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("/home/joe/Phi-3.5-MoE-instruct")

messages = [
    {
        "role": "user",
        "content": "Can you provide ways to eat combinations of bananas and dragonfruits?",
    }
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 128,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

# warmup
for _ in range(3):
    pipe(messages, **{
        "max_new_tokens": 1,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    })

torch.cuda.synchronize()
tic = time.time()
output = pipe(messages, **generation_args)
print(f"E2E latency: {(time.time() - tic):2f} secs")
print(output[0]["generated_text"])
