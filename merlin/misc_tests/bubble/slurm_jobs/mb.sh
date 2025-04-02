#!/bin/bash


# CMD="nsys profile \
#     --capture-range=cudaProfilerApi \
#     --capture-range-end=stop \
# CMD="torchrun \
#     --nnodes=$SLURM_JOB_NUM_NODES \
#     --nproc-per-node=$SLURM_GPUS_PER_NODE \
#     --rdzv_id $RANDOM \
#     --rdzv_backend c10d \
#     --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#     ./mixtral_8x7b_v0_h100.py \
#     --model-path /home/u20008787/merlin_mixtral_weights/v0 \
#     --prompt-path /home/u20008787/ntu_paslab_llm/mixtral/prompts/diverse_short.json \
#     --n-prompts 32 \
#     --batch-size 1 \
#     --max-tokens 40"


for ((bs = 1; bs <= 256; bs=bs*2))
do
    echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    echo "BATCH_SIZE=$bs, NODE_ID=$SLURM_NODEID"
    # nsys profile \
    # --capture-range=cudaProfilerApi \
    # --capture-range-end=stop \
    torchrun \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc-per-node=$SLURM_GPUS_PER_NODE \
        --node-rank=$SLURM_NODEID \
        --rdzv_id $RDZV_ID \
        --rdzv_backend c10d \
        --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
        /home/u20008787/ntu_paslab_llm/merlin/misc_tests/bubble/sync_latency.py \
            --batch-size $bs \
            --max-tokens 40 \

done