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
    prompt = f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}) {choice}\n"
    prompt += "Using only one token to answer (choose A, B, C, or D):"
    return prompt

def run_inference(model_path, prompt, max_tokens):
    """
    调用 solo_gpu_model.py 来运行推理，使用 subprocess 调用命令行
    """
    command = [
        "python", "solo_gpu_model.py", 
        "--model-path", model_path,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--n-prompts", "1" 
    ]
    
        # 使用 subprocess 运行命令，实时打印输出到控制台
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # 实时输出内容
    output_lines = []
    for line in process.stdout:
        print(line, end = '')  # 实时打印
        output_lines.append(line)  # 收集输出

    process.wait()
    #print(output_lines)
    clean_lines = [line.strip() for line in output_lines if line.strip()]
    print(clean_lines[-1][0])
    # 假设模型的输出为最后一行结果
    return clean_lines[-1][0]  # 返回最后一行结果，作为模型推理的输出

def evaluate_model(model_path, data_dir, split):
    datasets = load_mmlu_data(data_dir, split)
    total_correct = 0
    total_questions = 0

    with open("MMLU.txt", "w") as f:
        for subject, data in datasets.items():
            correct_predictions = 0
            f.write(f"{subject}\n")

            for i in range(data.shape[0]):
                question = data.iloc[i, 0]
                options = data.iloc[i, 1:5].values
                correct_answer = data.iloc[i, 5]

                prompt = format_prompt(question, options)
                
                # 调用推理函数
                predicted_answer = run_inference(model_path, prompt, 2)
                print("The answer is",predicted_answer)
                f.write(f"{predicted_answer}  ")
                # 检查预测结果是否与正确答案一致
                if predicted_answer == correct_answer:
                    correct_predictions += 1
                
                total_questions += 1
            total_correct += correct_predictions
            
            accuracy = correct_predictions / data.shape[0]
            print(f"{subject} Accuracy: {accuracy:.2%}")
            f.write(f"\n{subject} Accuracy: {accuracy:.2%}")
            f.write(f"\ncorrect_prediction: {correct_predictions}\n")
            f.write("=====================================================================\n\n")

        overall_accuracy = total_correct / total_questions
        print(f"Overall Accuracy: {overall_accuracy:.2%}")
        print("total_correct:",total_correct,"   ","total_questions",total_questions)
        f.write(f"Overall Accuracy: {overall_accuracy:.2%}\n")
        f.write(f"total_correct: {total_correct}  total_questions: {total_questions}\n")
    f.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="模型的路径")
    parser.add_argument("--data-dir", type=str, required=True, help="MMLU 数据集所在的目录")
    parser.add_argument("--split", type=str, help="使用的评估集")
    
    args = parser.parse_args()

    evaluate_model(args.model_path, args.data_dir, args.split)