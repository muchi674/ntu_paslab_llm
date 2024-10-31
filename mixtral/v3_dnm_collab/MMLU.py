import pandas as pd
import argparse
import os
import numpy as np
import subprocess  

choices = ["A", "B", "C", "D"]

def load_mmlu_data(data_dir, split):
    files = os.listdir(os.path.join(data_dir, split))
    datasets = {}
    
    for file in files:
        if file.endswith(".csv"):
            subject = file.split(f"_{split}.csv")[0]
            datasets[subject] = pd.read_csv(os.path.join(data_dir, split, file), header=None)
    
    return datasets

def format_prompt(question, choices):
    """
    格式化 MMLU 问题和选项为适合模型推理的 prompt
    """
    prompt = f"Answer the question for choosing one char from A, B, C, D. **DO NOT output other text than the correct result. ONLY generate one char for the answer**: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}  {choice}\n"
    prompt += "Answer:"
    return prompt

def run_inference(model_path, prompt, max_tokens):
    """
    调用 solo_gpu_model.py 来运行推理，使用 subprocess 调用命令行
    """
    command = [
        "python", "solo_gpu_model.py", 
        "--model-path", model_path,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens)
    ]
    
        # 使用 subprocess 运行命令，实时打印输出到控制台
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # 实时输出内容
    output_lines = []
    for line in process.stdout:
        print(line, end = '')  # 实时打印
        output_lines.append(line)   # 收集输出

    process.wait()
    #print(output_lines)
    clean_lines = [line.strip() for line in output_lines if line.strip()]
    #print(clean_lines[-1][0])
    # 假设模型的输出为最后一行结果
    return clean_lines[-1][0]  # 返回最后一行结果，作为模型推理的输出

def evaluate_model(model_path, data_dir, split):
    datasets = load_mmlu_data(data_dir, split)
    total_accuracy = 0
    total_questions = 0
    total_error_num = 0
    total_error_which = []
    total_correct_predictions_num = 0
    with open("MMLU.txt", "w") as f, open("../MMLU/dev_retest/top2_3.csv", "w") as f2:
        for subject, data in datasets.items():
            correct_predictions_num = 0
            error_num = 0
            error_which = []
            f.write(f"{subject}\n\n")

            for i in range(data.shape[0]):
                question = data.iloc[i, 0]
                options = data.iloc[i, 1:5].values
                correct_answer = data.iloc[i, 5]

                prompt = format_prompt(question, options)
                
                # 调用推理函数
                predicted_answer = run_inference(model_path, prompt, 2)
                print("The answer is",predicted_answer,"\n")
                f.write(f"{predicted_answer}  ")
                if predicted_answer not in choices:
                    error_num += 1
                    error_which.append(i)
                    data.iloc[[i]].to_csv(f2, index=False, header=False)

                # 检查预测结果是否与正确答案一致
                if predicted_answer == correct_answer:
                    correct_predictions_num += 1
                
                total_questions += 1
            
            total_correct_predictions_num += correct_predictions_num
            total_error_num += error_num
            accuracy = correct_predictions_num / data.shape[0]
            total_accuracy += accuracy
            print(f"{subject} Accuracy: {accuracy:.2%}")
            if error_num != 0:
                f.write(f"\nFuckkkk !!!! We have {error_num} errors, which are {error_which}\n")
                total_error_which.append((subject, error_which))
                
            
            f.write(f"\n{subject} Accuracy: {accuracy:.2%}\n")
            f.write(f"correct_predictions: {correct_predictions_num}\n")
            f.write("==========================================================================================================\n\n")

        overall_accuracy = total_accuracy / 57
        print(f"Overall Accuracy: {overall_accuracy:.2%}")
        print("total_questions",total_questions)
        f.write(f"Total error: {total_error_num}, which are {total_error_which}")
        f.write(f"\nOverall Accuracy: {overall_accuracy:.2%}\n")
        f.write(f"total_correct: {total_correct_predictions_num},  total_questions: {total_questions}\n")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="模型的路径")
    parser.add_argument("--data-dir", type=str, required=True, help="MMLU 数据集所在的目录")
    parser.add_argument("--split", type=str, help="使用的评估集")
    
    args = parser.parse_args()

    evaluate_model(args.model_path, args.data_dir, args.split)

    # python MMLU.py --model-path ~/Mixtral-8x7B-Instruct-v0.1-Official/ --data-dir ~/ntu_paslab_llm/mixtral/MMLU/ --split 