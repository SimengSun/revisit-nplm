#!/bin/bash

EXPERIMENT_PATH=/path/to/experiment
DATA_PATH=/path/to/data

python main.py \
  --action train \
  --model transformer \
  -v \
  --adaptive \
  --tie-weights \
  --tie-projs \
  --num-layers 16 \
  --num-heads 10 \
  --embedding-size 410 \
  --model-size 410 \
  --hidden-dim 2100  \
  --attn-type learned \
  --attn-impl full \
  -b 20 \
  --batch-length 512 \
  --bsz-gpu0 4 \
  -d $DATA_PATH \
  --split train \
  --accumulate-steps 1 \
  --max-checkpoints 1 \
  --label-smoothing 0 \
  --checkpoint-directory $EXPERIMENT_PATH \
  --max-steps 200000 \
  -l 0.00025 \
  --final-learning-rate 0 \
  --warmup-steps 4000
