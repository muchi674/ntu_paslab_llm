#!/bin/bash

which python

SIF=/home/u3114747/paslab_llm.sif
SINGULARITY="singularity run --nv $SIF"

PRL_MODEL_PATH="/home/u3114747/m822_parallel/pp-expert-attn-tp"
DESIGN_PATH="/home/u3114747/ntu_paslab_llm/merlin/partitioners/designs/8x22b-pp-expert-attn-tp.json"
PROMPT_PATH="/home/u3114747/ntu_paslab_llm/merlin/prompts/mixtral_8x7b_128.json"

N_PROMPTS=32
BATCH_SIZE=1
 echo "started running mixtral_8x7b_graph.py"
while [[ $N_PROMPTS -le 1024 && $BATCH_SIZE -le 32 ]]; do
    echo "N_PROMPTS: $N_PROMPTS, BATCH_SIZE: $BATCH_SIZE"

    $SINGULARITY torchrun \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc-per-node=$SLURM_GPUS_PER_NODE \
        --rdzv_id $RDZV_ID \
        --rdzv_backend c10d \
        --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
        /home/u3114747/ntu_paslab_llm/merlin/models/mixtral_8x7b_graph.py \
        --model-path $PRL_MODEL_PATH \
        --prompt-path $PROMPT_PATH \
        --n-prompts $N_PROMPTS \
        --batch-size $BATCH_SIZE \
        --max-tokens 128 \
        --hide-resp

    N_PROMPTS=$((N_PROMPTS * 2))
    BATCH_SIZE=$((BATCH_SIZE * 2))
done
