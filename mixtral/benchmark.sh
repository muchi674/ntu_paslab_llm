#!/usr/bin/bash
# for dir in "v0_poc" "v1_gpu_only" "v2_static_collab" "v3_dnm_collab"
for dir in "v2_static_collab"
do
    for batch_size in 1 2 4 8 16 32 64 128
    do
        echo "model version: $dir"
        echo "batch_size: $batch_size"
        python ./"$dir"/solo_gpu_model.py --model-path ~/Mixtral-8x7B-Instruct-v0.1-Official/ --prompt-path ./prompts/diverse_short.json --n-prompts ${batch_size} --max-tokens 128 --hide-resp
        sleep 120s
    done
    # sleep 300s
done