#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python attention_visualization.py \
--bert_model \
bert-base-uncased \
--batch_size \
128 \
--from_pretrained \
save/single_stream_baseline/best_cp.bin \
--config_file \
config/bert_base_6layer_6conect.json \
--task \
1 \
--split \
train