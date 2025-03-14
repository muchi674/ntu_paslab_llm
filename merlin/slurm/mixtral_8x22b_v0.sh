#!/bin/bash

#Batch Job Paremeters
#SBATCH --account=GOV113121
#SBATCH --partition=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1             # one torchrun per node https://stackoverflow.com/a/65897194
#SBATCH --cpus-per-gpu=1
#SBATCH --mail-type=END,BEGIN           # Send the mail when the job starts and finishes.
#SBATCH --mail-user=kyle355469@gmail.com
#SBATCH --time=00:10:00                 # total run time limit (HH:MM:SS)
# net
export UCX_NET_DEVICES=mlx5_0:1
export UCX_IB_GPU_DIRECT_RDMA=1

# enable NCCL log
# export NCCL_DEBUG=INFO

# setup master address and port
# https://iservice.nchc.org.tw/nchc_service/nchc_service_news_content.php?contentId=1000254&type=all_content&newsId=58494
# HPC容器環境打包技術與效能案例實作.pdf
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

CMD="nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc-per-node=$SLURM_GPUS_PER_NODE \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    ../models/mixtral_8x7b_v0_h100.py \
    --model-path /home/paslab504llm/v0 \
    --prompt-path /home/paslab504llm/ntu_paslab_llm/mixtral/prompts/diverse_short.json \
    --n-prompts 4 \
    --batch-size 1 \
    --max-tokens 16"

# SRUN_CMD="$SINGULARITY $CMD"

# https://discuss.pytorch.org/t/distributed-training-on-slurm-cluster/150417/8
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist = " $(scontrol show hostnames "$SLURM_JOB_NODELIST")
echo "Number of nodes = " $SLURM_JOB_NUM_NODES
echo "Ntasks per node = "  $SLURM_NTASKS_PER_NODE
echo "Cpus per task = " $SLURM_CPUS_PER_TASK
echo "Master =" $MASTER_ADDR $MASTER_PORT
echo "srun" $CMD
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

srun $CMD

# Ref: https://github.com/PrincetonUniversity/multi_gpu_training/tree/main/02_pytorch_ddp