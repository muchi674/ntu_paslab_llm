#!/bin/bash

which python

OG_MODEL_PATH="/home/paslab504llm/Mixtral-8x7B-Instruct-v0.1"
PRL_MODEL_PATH="/home/paslab504llm/m87_parallel/intra-ep-intra-tp-attn"
DESIGN_PATH="/home/paslab504llm/ntu_paslab_llm/merlin/partitioners/designs/ep-attn-tp.json"
PROMPT_PATH="/home/paslab504llm/ntu_paslab_llm/merlin/prompts/mixtral_8x7b_128.json"

mkdir $PRL_MODEL_PATH
cp $OG_MODEL_PATH/config.json $PRL_MODEL_PATH
cp $OG_MODEL_PATH/*token* $PRL_MODEL_PATH

echo "[`date`] started partitioning weights"
python /home/paslab504llm/ntu_paslab_llm/merlin/partitioners/mixtral_8x7b.py \
    --model-path $OG_MODEL_PATH \
    --design-path $DESIGN_PATH \
    --output-path $PRL_MODEL_PATH

N_PROMPTS=32
BATCH_SIZE=1

while [[ $N_PROMPTS -le 1024 && $BATCH_SIZE -le 32 ]]; do
    echo "N_PROMPTS: $N_PROMPTS, BATCH_SIZE: $BATCH_SIZE"

    torchrun \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc-per-node=$SLURM_GPUS_PER_NODE \
        --rdzv_id $RDZV_ID \
        --rdzv_backend c10d \
        --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
        /home/paslab504llm/ntu_paslab_llm/merlin/models/mixtral_8x7b_graph.py \
        --model-path $PRL_MODEL_PATH \
        --prompt-path $PROMPT_PATH \
        --n-prompts $N_PROMPTS \
        --batch-size $BATCH_SIZE \
        --max-tokens 128 \
        --hide-resp

    N_PROMPTS=$((N_PROMPTS * 2))
    BATCH_SIZE=$((BATCH_SIZE * 2))
done
