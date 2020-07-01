#!/usr/bin/env bash 
# -*- coding: utf-8 -*- 



REPO_PATH=/home/xiaoyli1110/xiaoya/Coref-tf
export PYTHONPATH="$PYTHONPATH:/home/xiaoyli1110/xiaoya/Coref-tf"
# export TPU_NAME=tensorflow-tpu
export TPU_NAME=tf-tpu
GCP_PROJECT=xiaoyli-20-04-274510
OUTPUT_DIR=gs://mention_proposal/output_1



python3 ${REPO_PATH}/run/train_mention_proposal.py \
--output_dir=${OUTPUT_DIR} \
--do_train=True \
--use_tpu=True \
--iterations_per_loop=500 \
--tpu_name=${TPU_NAME} \
--tpu_zone=us-central1-f \
--gcp_project=${GCP_PROJECT} \
--num_tpu_cores=1