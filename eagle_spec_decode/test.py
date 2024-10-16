import argparse
import os
import time
from pathlib import Path
from statistics import mean

import torch
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from EAGLE.eagle.model.ea_model import EaModel as OnChipEaModel
from gpu_only_offloading.eagle.model.ea_model import EaModel as OffloadedEaModel
from static_collab.eagle.model.ea_model import EaModel as StaticCollabEaModel


def vanilla(gpu: torch.device):
    model_path = "/home/joe/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=gpu,
    )
    return tokenizer, model


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
        total_token=59,
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
        draft_device=gpu,
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
        draft_device=gpu,
    )

    return tokenizer, model


def load_messages():
    datasetparent = "./data/pg19/"
    d_files = os.listdir(datasetparent)
    dataset = load_dataset(
        "json", data_files=[datasetparent + name for name in d_files], split="train"
    )

    # for testing
    [1010, 2020, 4040, 6840, 11725]

    messages = [
        {
            "role": "system",
            "content": dataset[0]["text"][:490],
        }
    ]

    return messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="vanilla")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")

    if args.strategy == "vanilla":
        tokenizer, model = vanilla(gpu)
    elif args.strategy == "on_chip":
        tokenizer, model = on_chip_test(gpu)
    elif args.strategy == "gpu_only_offloading":
        tokenizer, model = gpu_only_offloading_test(cpu, gpu)
    elif args.strategy == "static_collab":
        tokenizer, model = static_collab_test(cpu, gpu)
    else:
        raise RuntimeError("invalid decoding strategy")

    model.eval()
    print("FINISHED INITIALIZING MODEL")

    # messages = [
    #     {
    #         "role": "system",
    #         "content": "Imagine you are an experienced Ethereum developer tasked with creating a smart contract for a blockchain messenger. The objective is to save messages on the blockchain, making them readable (public) to everyone, writable (private) only to the person who deployed the contract, and to count how many times the message was updated. Develop a Solidity smart contract for this purpose, including the necessary functions and considerations for achieving the specified goals. Please provide the code and any relevant explanations to ensure a clear understanding of the implementation.",
    #     }
    # ]
    messages = load_messages()
    input_ids: torch.Tensor = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(gpu)
    print(f"NUM IN TOKENS: {input_ids.shape[1]}")

    latencies = []
    if args.strategy == "vanilla":
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        # warmup
        for _ in range(3):
            model.generate(
                input_ids,
                max_new_tokens=128,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.5,
            )

        torch.cuda.synchronize()

        while len(latencies) < 10:
            start_time = time.time()

            output_ids: torch.Tensor = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.5,
            )

            torch.cuda.synchronize()
            if output_ids.shape[1] - input_ids.shape[1] > args.max_new_tokens * 0.95:
                latencies.append(time.time() - start_time)

    else:
        # warmup
        for _ in range(3):
            model.eagenerate(
                input_ids, temperature=0.5, max_new_tokens=1, is_llama3=True
            )

        torch.cuda.synchronize()

        while len(latencies) < 10:
            start_time = time.time()

            output_ids = model.eagenerate(
                input_ids,
                temperature=0.5,
                max_new_tokens=args.max_new_tokens,
                max_length=4096,
                is_llama3=True,
                profile=True,
            )

            torch.cuda.synchronize()
            if output_ids.shape[1] - input_ids.shape[1] > args.max_new_tokens * 0.95:
                latencies.append(time.time() - start_time)

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    avg_lat = mean(latencies)
    print(f"NUM OUT TOKENS: {output_ids.shape[1] - input_ids.shape[1]}")
    print(f"AVG E2E LAT: {avg_lat:.2f}")
    print(latencies)
    print(f"AVG E2E THROUGHPUT t/s: {(args.max_new_tokens / avg_lat):.2f}")
    print(f"RESPONSE:")
    # print(output)
