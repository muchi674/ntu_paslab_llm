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

    with open("Answers_for_datasets.txt", "w") as f:
        f.write(f"Which set: {split}\n\n")
        for subject, data in datasets.items():
            f.write(f"{subject}:   ") 
            for i in range(data.shape[0]):

                correct_answer = data.iloc[i, 5]
                f.write(f"{correct_answer}   ")

            f.write("\n\n=====================================================================\n\n")

    f.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="MMLU 数据集所在的目录")
    parser.add_argument("--split", type=str, help="使用的评估集")
    
    args = parser.parse_args()

    dataset(args.data_dir, args.split)