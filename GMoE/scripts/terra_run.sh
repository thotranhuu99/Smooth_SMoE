#!/bin/bash
# ================================
# Script train GMOE
# ================================

DATASET="Terra"
TOPK=2
NUM_EXPERTS=6
# ROUTER="cosine_top"
ROUTER="smooth_top"
LR=5e-5
WEIGHT_DECAY=1e-4
SEED=0
TEST_ENV=0

OUTPUT_DIR="./results/${DATASET}/${ROUTER}_1/testenv_${TEST_ENV}-seed_${SEED}-lr_${LR}-w_${WEIGHT_DECAY}/"


# Export GPU
export CUDA_VISIBLE_DEVICES=3

# Training
python -m domainbed.scripts.train \
  --data_dir=./domainbed/data/ \
  --algorithm=GMOE \
  --dataset=${DATASET} \
  --test_env=${TEST_ENV} \
  --output_dir=${OUTPUT_DIR} \
  --hparams="{\"topk\": ${TOPK}, \"num_experts\": ${NUM_EXPERTS}, \"router\": \"${ROUTER}\", \"lr\": ${LR}, \"weight_decay\": ${WEIGHT_DECAY}}" \
  --seed=${SEED}
