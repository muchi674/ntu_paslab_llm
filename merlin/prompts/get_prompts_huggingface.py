from datasets import load_dataset
import json, os
from pathlib import Path
from tqdm import tqdm

# ① 下载/加载数据集
ds = load_dataset("THUDM/LongWriter-6k", split="train")

prompts = []
for row in tqdm(ds, desc="collect"):
    # LongWriter-6k 每条样本有一个 messages 字段:
    # messages 是 [{'role':'user','content':'...'}, ...]
    # 下面只取所有 content 并用换行拼接成一段文本
    text = "\n".join(m["content"] for m in row["messages"])
    prompts.append(text)

# ② 写出 json（和 mixtral_8x7b_128.json 同结构）
with open("longwriter_6k.json", "w", encoding="utf-8") as f:
    json.dump({"prompts": prompts}, f, ensure_ascii=False, indent=4)

print(f"已生成 {len(prompts)} 条样本 -> prompts/longwriter_6k.json")