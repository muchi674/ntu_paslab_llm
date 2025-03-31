#!/bin/bash

n_prompts=32
batch_size=1

while [[ $n_prompts -le 1024 && $batch_size -le 128 ]]; do
    echo "n_prompts: $n_prompts, batch_size: $batch_size"
    
    torchrun \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc-per-node=$SLURM_GPUS_PER_NODE \
        --rdzv_id $RDZV_ID \
        --rdzv_backend c10d \
        --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
        ../models/mixtral_8x7b_graph.py \
        --model-path /home/paslab504llm/2xh100-ep \
        --prompt-path /home/paslab504llm/ntu_paslab_llm/merlin/prompts/mixtral_8x7b_128.json \
        --n-prompts $n_prompts \
        --batch-size $batch_size \
        --max-tokens 128

    n_prompts=$((n_prompts * 2))
    batch_size=$((batch_size * 2))
done
