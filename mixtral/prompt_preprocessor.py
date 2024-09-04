import csv
import json

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

if __name__ == "__main__":
    tokenizer = MistralTokenizer.v1()
    prompts = []
    with open("./prompts/fka_prompts.csv", "r") as csvfile:
        dataset = csv.reader(csvfile)
        for idx, row in enumerate(dataset):
            if idx == 0:
                continue
            p = tokenizer.decode(
                tokenizer.encode_chat_completion(
                    ChatCompletionRequest(messages=[UserMessage(content=row[1])])
                ).tokens[4:-4][:128]
            ) # 4 is [/INST] token, 128 is target length
            prompts.append(p)

    with open('./diverse_short.json', 'w') as f:
        json.dump({"prompts": prompts}, f)
