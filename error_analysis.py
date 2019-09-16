import argparse
import json
import logging
import os
import random
from collections import Counter
from io import open
import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm
from bisect import bisect
import yaml
from easydict import EasyDict as edict
import pickle
import sys
import pdb

import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from vilbert.task_utils import LoadDatasetEval, LoadLosses, ForwardModelsTrain, ForwardModelsVal, EvaluatingModel

import vilbert.utils as utils
import torch.distributed as dist

def getErrors():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--from_pretrained",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--config_file",
        default="config/bert_config.json",
        type=str,
        help="The config file which specified the model details.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--do_lower_case",
        default=True,
        type=bool,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
             "0 (default value): dynamic loss scaling.\n"
             "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument(
        "--num_workers", type=int, default=10, help="Number of workers in the dataloader."
    )
    parser.add_argument(
        "--save_name",
        default='',
        type=str,
        help="save name for training.",
    )
    parser.add_argument(
        "--batch_size", default=1000, type=int, help="what is the batch size?"
    )
    parser.add_argument(
        "--tasks", default='', type=str, help="1-2-3... training task separate by -"
    )
    parser.add_argument(
        "--in_memory", default=False, type=bool, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--baseline", action="store_true", help="whether use single stream baseline."
    )
    parser.add_argument(
        "--split", default="", type=str, help="which split to use."
    )

    args = parser.parse_args()
    with open('vlbert_tasks.yml', 'r') as f:
        task_cfg = edict(yaml.safe_load(f))


    if args.baseline:
        from pytorch_pretrained_bert.modeling import BertConfig
        from vilbert.basebert import BaseBertForVLTasks
    else:
        from vilbert.vilbert import BertConfig
        from vilbert.vilbert import VILBertForVLTasks

    task_names = []
    for i, task_id in enumerate(args.tasks.split('-')):
        task = 'TASK' + task_id
        name = task_cfg[task]['name']
        task_names.append(name)

    # timeStamp = '-'.join(task_names) + '_' + args.config_file.split('/')[1].split('.')[0]
    timeStamp = args.from_pretrained.split('/')[1] + '-' + args.save_name
    savePath = os.path.join(args.output_dir, timeStamp)

    config = BertConfig.from_json_file(args.config_file)
    bert_weight_name = json.load(open("config/" + args.bert_model + "_weight_name.json", "r"))

    task_batch_size, task_num_iters, task_ids, task_datasets_val, task_dataloader_val \
        = LoadDatasetEval(args, task_cfg, args.tasks.split('-'))

    # ["img_id"] for indexing
    val_results = json.load(open(os.path.join('./results/best_finetune_checkpoint-/val_result.json'), 'r'))
    true_val_items = task_datasets_val['TASK1']._entries

    assert len(val_results) == len(true_val_items)

    errors = []
    for i in tqdm(range(len(val_results))):
        if np.asarray(val_results[i]['answer']).argmax() != true_val_items[i]['target']:
            errors.append(true_val_items[i])
    print(len(errors) / len(val_results))
    pickle.dump(errors, open(os.path.join('./results/best_finetune_checkpoint-/val_errors.pkl'), 'wb'))
    return errors

def countPointers(errors):
    pointer_stats = Counter()
    for error in errors:
        question = error['question']
        num_pointers = 0
        for word in question:
            if type(word) is list:
                num_pointers += len(word)
        pointer_stats[num_pointers] += 1
    return pointer_stats

if __name__ == '__main__':
    pre_extract_errors = True
    if not pre_extract_errors:
        errors = getErrors()
    else:
        errors = pickle.load(open(os.path.join('./results/best_finetune_checkpoint-/val_errors.pkl'), 'rb'))

    pointerCounters = countPointers(errors)
    trunkCounter = Counter()
    geq_5 = 0
    for key, value in pointerCounters.items():
        if key > 4:
            geq_5 += value
        else:
            trunkCounter[key] = value
    trunkCounter[5] = geq_5
    print(trunkCounter)
    print(pointerCounters)
