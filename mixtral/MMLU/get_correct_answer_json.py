import pandas as pd
import argparse
import os
import numpy as np
import subprocess  


def load_mmlu_data(data_dir, split):
    files = os.listdir(os.path.join(data_dir, split))
    datasets = {}
    
    for file in files:
        if file.endswith(".csv"):
            subject = file.split(f"_{split}.csv")[0]
            datasets[subject] = pd.read_csv(os.path.join(data_dir, split, file), header=None)
    
    return datasets

def dataset(data_dir, split):
    datasets = load_mmlu_data(data_dir, split)
    total_questions_num = 0
    num_which_are = []

    with open("Answers_for_datasets.txt", "w") as f:
        f.write(f"Which set: {split}\n\n")
        for subject, data in datasets.items():
            f.write(f"{subject}:   ") 
            questions_num = 0
            for i in range(data.shape[0]):
                correct_answer = data.iloc[i, 5]
                f.write(f"{correct_answer}   ")
            num_which_are.append([subject, questions_num])
        f.write(f"\nTotal {total_questions_num} questions, which are:\n")
        for i in range(len(num_which_are)):
            f.write(f"{num_which_are[i]}\n")

    f.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="MMLU 数据集所在的目录")
    parser.add_argument("--split", type=str, help="使用的评估集")
    
    args = parser.parse_args()

    dataset(args.data_dir, args.split)
    
    # python get_correct_answer.py --data-dir  ~/ntu_paslab_llm/mixtral/MMLU --split