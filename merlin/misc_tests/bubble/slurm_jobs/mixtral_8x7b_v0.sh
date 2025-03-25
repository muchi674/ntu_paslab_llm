#!/bin/bash


# setup master address and port
# https://iservice.nchc.org.tw/nchc_service/nchc_service_news_content.php?contentId=1000254&type=all_content&newsId=58494
# HPC容器環境打包技術與效能案例實作.pdf
# nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
# nodes_array=($nodes)
# head_node=${nodes_array[0]}
# export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

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
#     --max-tokens 40 \
#     --hide-resp"

for ((bs = 1; bs <= 256; bs=bs*2))
do
    echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    echo "BATCH_SIZE=$bs"
    torchrun \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc-per-node=$SLURM_GPUS_PER_NODE \
        --rdzv_id $RANDOM \
        --rdzv_backend c10d \
        --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
        /home/u20008787/ntu_paslab_llm/merlin/misc_tests/bubble/mixtral_8x7b_v0_h100.py \
            --model-path /home/u20008787/merlin_mixtral_weights/v0 \
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