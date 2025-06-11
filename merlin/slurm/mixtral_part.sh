#!/bin/bash

which python

SIF=/home/u3114747/paslab_llm.sif
SINGULARITY="singularity run --nv $SIF"

OG_MODEL_PATH="/home/u3114747/Mixtral-8x22B-Instruct-v0.1"
PRL_MODEL_PATH="/home/u3114747/m822_parallel/ep-expert-attn-tp"
DESIGN_PATH="/home/u3114747/ntu_paslab_llm/merlin/partitioners/designs/8x22b-ep-expert-attn-tp.json"

mkdir $PRL_MODEL_PATH
cp $OG_MODEL_PATH/config.json $PRL_MODEL_PATH
cp $OG_MODEL_PATH/*token* $PRL_MODEL_PATH

# echo "[`date`] started partitioning weights"
# python /home/paslab504llm/ntu_paslab_llm/merlin/partitioners/mixtral_8x7b.py \
#     --model-path $OG_MODEL_PATH \
#     --design-path $DESIGN_PATH \
#     --output-path $PRL_MODEL_PATH

echo "[`date`] started partitioning mixtral weights"
$SINGULARITY python /home/u3114747/ntu_paslab_llm/merlin/partitioners/mixtral_8x22b.py \
    --model-path $OG_MODEL_PATH \
    --design-path $DESIGN_PATH \
    --output-path $PRL_MODEL_PATH \
    --n-part 2
