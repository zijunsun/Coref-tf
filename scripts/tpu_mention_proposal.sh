#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



# REPO_PATH=/home/lixiaoya/xiaoy_tf
REPO_PATH=/home/xiaoyli1110/xiaoya/Coref-tf
export PYTHONPATH=${REPO_PATH}
export PYTHONPATH="$PYTHONPATH:/home/xiaoyli1110/xiaoya/Coref-tf"
export TPU_NAME=tensorflow-tpu

export STORAGE_BUCKET=gs://xiaoy-data
export MODEL_DIR=$STORAGE_BUCKET/corefqa

output_dir=gs://corefqa-output/2020-06-10/spanbert-base 
GCP_PROJECT=xiaoyli-20-04-274510
DATA_DIR=$STORAGE_BUCKET/data
TRAIN_FILE=${DATA_DIR}/train.english.tfrecord

OUTPUT_DIR=/home/lixiaoya/finetune_output 
TRAIN_FILE=/xiaoya/data/train.english.tfrecord


# CUDA_VISIBLE_DEVICES=0 
python3 ${REPO_PATH}/run/train_mention_proposal.py \
--output_dir=${OUTPUT_DIR} \
--do_train=True \
--use_tpu=True \
--tpu_name=${TPU_NAME} \
--gcp_project=${GCP_PROJECT} \
--tpu_zone=us-central1-f \
--num_tpu_cores 1