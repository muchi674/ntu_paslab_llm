import csv
import json

def csv_to_llm_prompts(csv_file, json_file):
    prompts = []
    
    # 读取 CSV 文件
    with open(csv_file, mode='r') as f:
        csv_reader = csv.reader(f)
        
        
        for row in csv_reader:
            
            question_data = {
                "question": row[0],
                "options": row[1:5],  # 选项（排除正确答案）
            }
            
            
            prompt = f"Question: {question_data['question']}\n"
            prompt += "\n".join([f"{chr(65+i)}  {opt}" for i, opt in enumerate(question_data['options'])])
            prompt += "\nGive me your answer first!!"
            
            
            prompts.append(prompt)

    data_to_save = {"prompts": prompts}
    with open(json_file, mode='w') as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=True)


csv_to_llm_prompts('dev_retest/top2_3.csv', 'dev_retest/top2_3.json')








'''
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

def csv_to_llm_prompts(csv_file, json_file):
    prompts = []
    
    # 读取 CSV 文件
    with open(csv_file, mode='r') as f:
        csv_reader = csv.reader(f)
        
        
        for row in csv_reader:
            
            question_data = {
                "question": row[0],
                "options": row[1:5],  # 选项（排除正确答案）
            }
            
            
            prompt = f"Question: {question_data['question']}\n"
            prompt += "\n".join([f"{chr(65+i)}  {opt}" for i, opt in enumerate(question_data['options'])])
            prompt += "\nGive me your answer first!!"
            
            
            prompts.append(prompt)

    data_to_save = {"prompts": prompts}
    with open(json_file, mode='w') as f:
        json.dump(prompts, f, indent=4, ensure_ascii=True)


csv_to_llm_prompts('dev_retest/top2_3.csv', 'dev_retest/top2_3.json')

'''