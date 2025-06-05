#!/bin/bash

n_prompts=1024
batch_size=32

while [[ $n_prompts -le 1024 && $batch_size -le 32 ]]; do

    jq ".shared_exec_args.n_prompts = $n_prompts" launch_config.json > tmp.$$.json && mv tmp.$$.json launch_config.json
    jq ".shared_exec_args.batch_size = $batch_size" launch_config.json > tmp.$$.json && mv tmp.$$.json launch_config.json

    echo "n_prompts: $n_prompts, batch_size: $batch_size"
    
    python launch_nodes.py --launch-config ./launch_config.json

    sleep 10m
    n_prompts=$((n_prompts * 2))
    batch_size=$((batch_size * 2))
done
