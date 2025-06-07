#!/bin/bash

#Batch Job Paremeters
#SBATCH --account=GOV113121
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1             # one torchrun per node https://stackoverflow.com/a/65897194
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00                 # total run time limit (HH:MM:SS)

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
export OMP_NUM_THREADS=8
export RDZV_ID=$RANDOM

cd /home/paslab504llm/ntu_paslab_llm/merlin
source .venv/bin/activate

OG_MODEL_PATH="/home/paslab504llm/Mixtral-8x22B-Instruct-v0.1"
PRL_MODEL_PATH="/home/paslab504llm/m822_parallel/inter-ep"
DESIGN_PATH="/home/paslab504llm/ntu_paslab_llm/merlin/partitioners/designs/ep.json"
# PROMPT_PATH="/home/paslab504llm/ntu_paslab_llm/merlin/prompts/mixtral_8x7b_128.json"

mkdir $PRL_MODEL_PATH
cp $OG_MODEL_PATH/config.json $PRL_MODEL_PATH
cp $OG_MODEL_PATH/*token* $PRL_MODEL_PATH

echo "[`date`] started partitioning weights"
CMD="python /home/paslab504llm/ntu_paslab_llm/merlin/partitioners/mixtral_8x22b.py \
    --model-path $OG_MODEL_PATH \
    --design-path $DESIGN_PATH \
    --output-path $PRL_MODEL_PATH \
    --n-part 2"

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