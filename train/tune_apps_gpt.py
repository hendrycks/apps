"""
Tune LM on Code
"""

import io
import logging
import math
import os
import pprint
import sys
import time
import json

import transformers

from tqdm import tqdm
from datasets import load_dataset
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from dataset_lm.base_lm_dataset import BaseLMDataset
from dataset_apps.APPSBaseDataset import APPSBaseDataset
from CustomTensorboardCallback import CustomTensorBoardCallback

# torch.set_num_threads(2)

# https://github.com/pytorch/pytorch/issues/11201
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def run_training(args, train_data):

    ## Checkpoint Loading ######################################################## 
    if args.load:
        if '2700' in args.load:
            model = transformers.GPTNeoForCausalLM.from_pretrained(args.load)
        else:
            model = transformers.GPT2LMHeadModel.from_pretrained(args.load)
        print(f"Loaded model from {args.load}")
    else:
        if "EleutherAI" in args.arch:
            model = transformers.GPTNeoForCausalLM.from_pretrained(args.arch)
        else:
            model = transformers.GPT2LMHeadModel.from_pretrained(args.arch)

    if args.resume:
        raise NotImplementedError
        model = transformers.GPT2LMHeadModel.from_pretrained(args.resume)
        print(f"Loaded model from {args.resume}")
        start_epoch = 0
        start_iteration = int(args.resume.split("-")[-1])
        print("start_iteration = ", start_iteration)
    else:
        start_iteration = 0

    ## Dataloading ######################################################## 
    train_data.start_iteration = start_iteration

    ## Start Loop ########################################################
    print(f"Starting main loop")

    training_args = transformers.TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        do_eval=False,
        do_predict=True,
        evaluation_strategy='no',
        eval_steps=0, 

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.05,
        # warmup_steps=args.lr_warmup_steps,
        # max_grad_norm=100000.0,

        logging_dir=args.save_dir, 
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_steps=args.save_freq,
        save_total_limit=2,

        dataloader_drop_last=True,
        dataloader_num_workers=3,

        local_rank=args.local_rank,

        deepspeed=args.deepspeed,
        fp16=args.fp16,
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )
    trainer.remove_callback(transformers.integrations.TensorBoardCallback)
    trainer.add_callback(CustomTensorBoardCallback())

    trainer.train()
    
    if args.local_rank == 0:
        model.save_pretrained(os.path.join(args.save_dir, "final_checkpoint"))


def get_dataset(args): 
    
    fnames = os.listdir(args.apps_train_files)
 
    train_data = APPSBaseDataset(
        dataroot=args.apps_dataroot, 
        problem_dirs=fnames,
        mode=args.arch, 
        max_tokens=2048 if ('EleutherAI' in args.arch or '2700' in args.load) else 1024,
        sample_mode=args.apps_sample_mode
    )

    return train_data


def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    os.makedirs(args.save_dir, exist_ok=True)
    
    train_data = get_dataset(args)

    # Save command to file
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    run_training(args, train_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Language Modelling on Code")
    parser.add_argument('--arch', default='gpt2', choices=transformers.GPT2_PRETRAINED_MODEL_ARCHIVE_LIST + ["EleutherAI/gpt-neo-2.7B"])
    parser.add_argument('--dummy-model', action='store_true')
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--resume', default=None, type=str)

    # Dataloading
    parser.add_argument('--apps-dataroot', default='../apps/', type=str)
    parser.add_argument('--apps-train-files', default='../apps/data_split/train.json', type=str)
    parser.add_argument('--apps-sample-mode', default='uniform_sol')
    
    # Training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    # parser.add_argument('--lr-warmup-steps', default=500, type=int)
    parser.add_argument('--batch-size-per-replica', default=8, type=int)
    parser.add_argument('--grad-acc-steps', default=4, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')

    # Logging and stuff
    parser.add_argument('--save-dir', default="checkpoints/TEMP", type=str)
    parser.add_argument('--log-freq', default=5, type=int)
    parser.add_argument('--save-freq', default=200, type=int)

    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%m-%d-%Y__%H:%M:%S"))
    
    main(args)
