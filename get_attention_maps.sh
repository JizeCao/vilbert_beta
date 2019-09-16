#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 attention_visualization.py \
--bert_model \
bert-base-uncased \
--batch_size \
32 \
--from_pretrained \
save/single_stream_baseline/best_cp.bin \
--config_file \
config/bert_base_baseline.json \
--task \
1 \
--split \
train \
--baseline