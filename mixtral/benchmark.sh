#!/usr/bin/bash
for batch_size in 1 2 4 8 16 32
do
    echo "batch_size: $batch_size"
    python ./v2_gpu_only/solo_gpu_model.py --model-path ~/Mixtral-8x7B-Instruct-v0.1-Official/ --prompt-path ./prompts/short.json --n-prompts ${batch_size} --max-tokens 128 --hide-resp
    sleep 120s
done