#!/usr/bin/env bash

#python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 train_tasks.py \
#--bert_model bert-base-uncased \
#--from_pretrained bert-base-uncased \
#--config_file config/bert_base_baseline.json \
#--learning_rate 2e-5 \
#--num_workers 16 \
#--tasks 1-2 \
#--save_name bert_base_baseline \
#--baseline \
#--fp16

#python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 train_tasks.py \
#--bert_model bert-base-uncased \
#--from_pretrained save/bert_base \
#--config_file config/bert_base_baseline.json \
#--learning_rate 2e-5 \
#--num_workers 16 \
#--tasks 1-2 \
#--save_name bert_base_with_zero_out_pt \
#--baseline \
#--fp16


python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 train_tasks.py \
--bert_model bert-base-uncased \
--from_pretrained save/bert_base_6_layer_6_connect/pytorch_model_9.bin \
--config_file config/bert_base_6layer_6conect.json \
--learning_rate 2e-5 \
--num_workers 16 \
--tasks 1-2 \
--save_name benchmark_16_with_pt \
--fp16