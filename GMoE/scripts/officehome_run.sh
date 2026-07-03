#!/bin/bash
# ================================
# Script train GMOE
# ================================

DATASET="OfficeHome"
TOPK=2
NUM_EXPERTS=6
# ROUTER="cosine_top"
ROUTER="smooth_top"
LR=1e-5
WEIGHT_DECAY=1e-6
SEED=4
TEST_ENV=0

OUTPUT_DIR="./results/${DATASET}/test-env_${TEST_ENV}/${TOPK}_${NUM_EXPERTS}_${ROUTER}_${SEED}/"


# Export GPU
export CUDA_VISIBLE_DEVICES=6

# Training
python -m domainbed.scripts.train \
  --data_dir=./domainbed/data/ \
  --algorithm=GMOE \
  --dataset=${DATASET} \
  --test_env=${TEST_ENV} \
  --output_dir=${OUTPUT_DIR} \
  --hparams="{\"topk\": ${TOPK}, \"num_experts\": ${NUM_EXPERTS}, \"router\": \"${ROUTER}\", \"lr\": ${LR}, \"weight_decay\": ${WEIGHT_DECAY}}" \
  --seed=${SEED}
