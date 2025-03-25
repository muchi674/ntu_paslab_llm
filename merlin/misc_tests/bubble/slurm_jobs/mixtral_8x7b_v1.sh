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
    echo "BATCH_SIZE=$bs"
    torchrun \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc-per-node=$SLURM_GPUS_PER_NODE \
        --node-rank=$SLURM_NODEID \
        --rdzv_id $RANDOM \
        --rdzv_backend c10d \
        --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
        /home/u20008787/ntu_paslab_llm/merlin/misc_tests/bubble/mixtral_8x7b_v1_h100.py \
            --model-path /home/u20008787/merlin_mixtral_weights/v1-n2-d2-4-4 \
            --prompt-path /home/u20008787/ntu_paslab_llm/mixtral/prompts/diverse_short.json \
            --n-prompts $((32 * bs)) \
            --batch-size $bs \
            --max-tokens 40 \
            --hide-resp
done

# SRUN_CMD="$SINGULARITY $CMD"

# https://discuss.pytorch.org/t/distributed-training-on-slurm-cluster/150417/8
# echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
# echo "Nodelist = " $(scontrol show hostnames "$SLURM_JOB_NODELIST")
# echo "Number of nodes = " $SLURM_JOB_NUM_NODES
# echo "Ntasks per node = "  $SLURM_NTASKS_PER_NODE
# echo "Cpus per task = " $SLURM_CPUS_PER_TASK
# echo "Master =" $MASTER_ADDR $MASTER_PORT
# echo "srun" $CMD
# echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# srun $CMD

# Ref: https://github.com/PrincetonUniversity/multi_gpu_training/tree/main/02_pytorch_ddp