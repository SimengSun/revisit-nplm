#!/bin/bash

EXPERIMENT_PATH=/path/to/experiment
DATA_PATH=/path/to/data

python main.py \
  --action train \
  --model nplm \
  --TFN \
  -v \
  --adaptive \
  --tie-projs \
  --tie-weights \
  --num-layers 16 \
  --num-heads 10 \
  --embedding-size 410 \
  --model-size 410 \
  --hidden-dim 2100  \
  --context-config 5 506 \
  --num-global-agg 1 \
  --global-aggregate kernel \
  -b 20 \
  --batch-length 512 \
  --bsz-gpu0 4 \
  -d $DATA_PATH \
  --split train \
  --accumulate-steps 1 \
  --label-smoothing 0 \
  --checkpoint-directory $EXPERIMENT_PATH \
  --max-checkpoints 1 \
  --max-steps 200000 \
  -l 0.00025 \
  --final-learning-rate 0 \
  --warmup-steps 4000 
