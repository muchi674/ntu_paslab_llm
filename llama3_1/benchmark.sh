#!/usr/bin/bash
for batch_size in 1 2 4 8 16 32
do
    for max_gen_len in 32 64 128
    do
        # echo "model version: $dir"
        echo "batch_size: $batch_size"
        python ./v0_cpu_only/model.py --model-path ~/Meta-Llama3.1-70B-Instruct/ --prompt-path ../mixtral/prompts/diverse_short.json --n-prompts ${batch_size} --fixed-prompt-len 64 --max-gen-len ${max_gen_len} --hide-resp
        sleep 180s
    done
done
