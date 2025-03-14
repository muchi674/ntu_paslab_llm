"""This script is currently only good for Mixtral8x7b"""
from pathlib import Path
import argparse
import json

from datasets import load_dataset
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


def cut_to_length(tokenizer, prompt: str, target_len: int) -> tuple[int, str]:
    tokens = tokenizer.encode_chat_completion(
        ChatCompletionRequest(messages=[UserMessage(content=prompt)])
    ).tokens
    # WARNING: this is hard-coded
    tokens = tokens[4:-4]
    cut = tokens[:target_len]
    return len(cut), tokenizer.decode(cut)

def main(n_samples: int, target_len: int, dest: str):
    dataset = load_dataset("Open-Orca/OpenOrca", split="train")
    tokenizer = MistralTokenizer.v1()
    verified = []

    i = 0
    while n_samples > 0:
        prompt = dataset[i]["question"]
        cut_len, cut = cut_to_length(tokenizer, prompt, target_len)
        if cut_len == target_len:
            verified.append(cut)
            n_samples -= 1
        i += 1 # purposedly allowing error out


    with open(Path(dest), 'w', encoding='utf-8') as f:
        json.dump({"prompts": verified}, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--target-len", type=int)
    parser.add_argument("--dest", type=str)
    args = parser.parse_args()

    main(
        args.n_samples,
        args.target_len,
        args.dest
    )
