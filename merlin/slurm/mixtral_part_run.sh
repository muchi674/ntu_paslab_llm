#!/bin/bash

which python

SIF=/home/u3114747/paslab_llm.sif
SINGULARITY="singularity run --nv $SIF"

PRL_MODEL_PATH="/home/u3114747/m822_parallel/inter-tp-intra-attn"
DESIGN_PATH="/home/u3114747/ntu_paslab_llm/merlin/partitioners/designs/8x22b-inter-tp-intra-attn.json"
PROMPT_PATH="/home/u3114747/ntu_paslab_llm/merlin/prompts/mixtral_8x7b_128.json"

echo $PRL_MODEL_PATH
echo "LOCAL_RANK: $LOCAL_RANK"
N_PROMPTS=512
BATCH_SIZE=16
 echo "started running mixtral_8x7b_graph.py"
PROFILE_NAME="nsys_output_n512_b16.qdrep"

# if [[ $SLURM_PROCID == 0 ]]; then
    nsys profile \
        --output "/home/u3114747/8x22_profile_results/${PROFILE_NAME}" \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --sample=none \
        --trace=cuda,nvtx,osrt \
        torchrun \
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
        --max-tokens 32 \
        --hide-resp
# else
#     $SINGULARITY torchrun \
#         --nnodes=$SLURM_JOB_NUM_NODES \
#         --nproc-per-node=$SLURM_GPUS_PER_NODE \
#         --rdzv_id $RDZV_ID \
#         --rdzv_backend c10d \
#         --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
#         /home/u3114747/ntu_paslab_llm/merlin/models/mixtral_8x7b_graph.py \
#         --model-path $PRL_MODEL_PATH \
#         --prompt-path $PROMPT_PATH \
#         --n-prompts $N_PROMPTS \
#         --batch-size $BATCH_SIZE \
#         --max-tokens 32 \
#         --hide-resp
# fi