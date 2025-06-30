#!/bin/bash

which python

OG_MODEL_PATH="/home/paslab504llm/Mixtral-8x22B-Instruct-v0.1"
PRL_MODEL_PATH="/home/paslab504llm/m822_parallel/experts-attn-tp"
DESIGN_PATH="/home/paslab504llm/ntu_paslab_llm/merlin/partitioners/designs/ep-experts-attn-tp.json"
PROMPT_PATH="/home/paslab504llm/ntu_paslab_llm/merlin/prompts/mixtral_8x7b_128.json"

echo $PRL_MODEL_PATH

# mkdir $PRL_MODEL_PATH
# cp $OG_MODEL_PATH/config.json $PRL_MODEL_PATH
# cp $OG_MODEL_PATH/*token* $PRL_MODEL_PATH

# echo "[`date`] started partitioning weights"
# python /home/paslab504llm/ntu_paslab_llm/merlin/partitioners/mixtral_8x7b.py \
#     --model-path $OG_MODEL_PATH \
#     --design-path $DESIGN_PATH \
#     --output-path $PRL_MODEL_PATH

# echo "[`date`] started partitioning weights"
# python /home/paslab504llm/ntu_paslab_llm/merlin/partitioners/mixtral_8x22b.py \
#     --model-path $OG_MODEL_PATH \
#     --design-path $DESIGN_PATH \
#     --output-path $PRL_MODEL_PATH \
#     --n-part 2

# N_PROMPTS=128
# BATCH_SIZE=4

# while [[ $N_PROMPTS -le 256 && $BATCH_SIZE -le 8 ]]; do
#     echo "N_PROMPTS: $N_PROMPTS, BATCH_SIZE: $BATCH_SIZE"

#     torchrun \
#         --nnodes=$SLURM_JOB_NUM_NODES \
#         --nproc-per-node=$SLURM_GPUS_PER_NODE \
#         --rdzv_id $RDZV_ID \
#         --rdzv_backend c10d \
#         --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#         /home/paslab504llm/ntu_paslab_llm/merlin/models/mixtral_8x7b_graph.py \
#         --model-path $PRL_MODEL_PATH \
#         --prompt-path $PROMPT_PATH \
#         --n-prompts $N_PROMPTS \
#         --batch-size $BATCH_SIZE \
#         --max-tokens 128 \
#         --hide-resp

#     N_PROMPTS=$((N_PROMPTS * 2))
#     BATCH_SIZE=$((BATCH_SIZE * 2))
# done

# nsys profile \
#     --capture-range=cudaProfilerApi \
#     --capture-range-end=stop \
#     --cuda-graph-trace=node \
#     --force-overwrite true \
#     -o m822_waiting_graph \
#     torchrun \
#         --nnodes=$SLURM_JOB_NUM_NODES \
#         --nproc-per-node=$SLURM_GPUS_PER_NODE \
#         --rdzv_id $RDZV_ID \
#         --rdzv_backend c10d \
#         --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#         /home/paslab504llm/ntu_paslab_llm/merlin/models/mixtral_8x7b_graph.py \
#         --model-path $PRL_MODEL_PATH \
#         --prompt-path $PROMPT_PATH \
#         --n-prompts 4 \
#         --batch-size 1 \
#         --max-tokens 8 \

torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc-per-node=$SLURM_GPUS_PER_NODE \
    --rdzv_id $RDZV_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    /home/paslab504llm/ntu_paslab_llm/merlin/models/mixtral_8x7b_graph.py \
    --model-path $PRL_MODEL_PATH \
    --prompt-path $PROMPT_PATH \
    --n-prompts 32 \
    --batch-size 1 \
    --max-tokens 128 \
