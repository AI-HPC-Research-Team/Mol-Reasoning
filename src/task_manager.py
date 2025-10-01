# -*- coding: utf-8 -*-
import logging
logger = logging.getLogger(__name__)

import warnings
#warnings.filterwarnings("ignore", message="A decoder-only architecture is being used, but right-padding was detected!")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.dataset_manager import Mol_predict_Dataset, Mol_reason_Dataset, MolDataset

from models.model_manager import MainModel

from torch.optim.lr_scheduler import StepLR
import datetime
from transformers import AutoTokenizer
import time

from utils import *

from accelerate import Accelerator
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
def train_epochs(train_loader, valid_loader, test_loader, model, optimizer, scheduler, args, best_loss = None):
    running_loss = AverageMeter()
    step = 0
    loss_values = {"train_loss": [], "val_loss": [], "test_loss": []}
    last_ckpt_file = None
    patience = 0
    device = args.device
    for epoch in range(args.epochs):
        logger.info("========Epoch %d========" % (epoch + 1))
        logger.info("Training...")
        #model.train()
        train_loss = []
        train_loader = tqdm(train_loader, desc="Training")
        for mol in train_loader:
            mol = ToDevice(mol, args.device)
            outputs = model(mol)
            #outputs = model(mol, stage = 'inference')
            loss = outputs['loss']
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            running_loss.update(loss.detach().cpu().item())
            step += 1
            if step % args.logging_steps == 0:
                logger.info("Steps=%d Training Loss=%.4lf" % (step, running_loss.get_average()))
                train_loss.append(running_loss.get_average())
                running_loss.reset()
        loss_values["train_loss"].append(np.mean(train_loss))
        val_loss = val_epochs(valid_loader, model, device)
        test_loss = val_epochs(test_loader, model, device)
        loss_values["val_loss"].append(val_loss)
        loss_values["test_loss"].append(test_loss)

        if best_loss == None or val_loss<best_loss :
            patience = 0
            best_loss = val_loss
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            if not os.path.exists(f"{args.ckpt_output_path}"):
                os.makedirs(f"{args.ckpt_output_path}")

            if last_ckpt_file is not None and os.path.exists(last_ckpt_file):
                os.remove(last_ckpt_file)
                print(f"Deleted checkpoint file: {last_ckpt_file}")
                
            ckpt_file = f"{epoch}_{timestamp}.pth"
            ckpt_path = os.path.join(f"{args.ckpt_output_path}", ckpt_file)
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_loss': best_loss
            }, ckpt_path)
                
            message = f"epoch: {epoch}, best_loss:{best_loss} ,val_loss:{val_loss}, {ckpt_file} saved. "
            print(message)

            last_ckpt_file = ckpt_path
            print(loss_values)
        else:
            patience = patience+1
            print(loss_values)
        if patience > args.patience:
            break
    return model, best_loss, optimizer, scheduler


def val_epochs(valid_loader, model, device):
    model.eval()
    val_loss = 0
    logger.info("Validating...")
    with torch.no_grad():
        valid_loader = tqdm(valid_loader, desc="Validation")
        for i, mol in enumerate(valid_loader):
            #print(mol)
            truth = mol['truth']
            del mol['truth']
            mol = ToDevice(mol, device)
            loss = model(mol)['loss']
            if(i==1):
                result = model.generate(mol)
                temp_result = result[0]
                print(f"truth:{truth[0]} | Result : {temp_result}")
            val_loss += loss.detach().cpu().item()
        logger.info("validation loss %.4lf" % (val_loss / len(valid_loader)))
    return val_loss / len(valid_loader)


def test_epochs(test_loader, model, device, message = None):
    model.eval()
    test_loss = 0
    logger.info("Testing...")
    i = 0
    with torch.no_grad():
        test_loader = tqdm(test_loader, desc="Test")
        id_list = []
        truth_list = []
        task_list = []
        result_list = []
        for data in test_loader:
            id = data['id']
            task = data['task']
            truth = data['truth']
            task_list = task_list + task
            truth_list = truth_list + truth
            id_list = id_list + id
            del data['truth']
            data = ToDevice(data, device)
            result = model.generate(data)
            if(i==1):
                temp_result = result[0]
                print(f"truth:{truth[0]} |Result : {temp_result}")
            i=i+1
            for r in result:
                result_list.append(r)
        
        return id_list, task_list, truth_list, result_list


#toy_mol_train_rot_data.csv
def add_arguments(parser):
    """

    """
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task", type=str, default=['mol_cap'])
    parser.add_argument("--dataset_path", type=str, default='../data/predict_data')
    parser.add_argument("--dataset_name", type=str, default='predict_dataset.csv')
    parser.add_argument("--ckpt_output_path", type=str, default="../ckpts/finetune_ckpts")
    parser.add_argument("--model_name", type=str, default='qwen_7b')
    parser.add_argument("--model_output_path", type=str, default="output")
    parser.add_argument("--log_save_path", type=str, default="log")
    parser.add_argument("--result_save_path", type=str, default="../result")
    parser.add_argument("--latest_checkpoint", type=str, default="../ckpts/finetune_ckpts")
    parser.add_argument("--model_pretrain", type=str, default="../ckpts/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--peft", type=str, default='lora')



if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    print(args)
        
    if(args.mode == 'train'):
        args.latest_checkpoint = None
        best_loss = None
        latest_checkpoint = args.latest_checkpoint
        if args.latest_checkpoint:
            print(f"Latest checkpoint: {latest_checkpoint}")
        else:
            print("No checkpoint found.")
        # dataset
        logger.info("Loading dataset ......")

        model = MainModel(args)

        state_dict = torch.load(latest_checkpoint, map_location='cpu', weights_only=True)["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
        
        print_model_info(model,level=4)
        tokenizer_org = model.tokenizer

        train_dataset = Mol_predict_Dataset(split = "train",
                                  tokenizer_org = tokenizer_org,
                                  args = args)
        valid_dataset = Mol_predict_Dataset(split = "valid",
                                  tokenizer_org = tokenizer_org,
                                  args = args)
        test_dataset = Mol_predict_Dataset(split = "test",
                                  tokenizer_org = tokenizer_org,
                                  args = args)
        logger.info("Loading dataset successed")

        logger.info("Loading dataloader ......")
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                  pin_memory=True)

        valid_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=args.num_workers,
                                pin_memory=True)
        test_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
        logger.info("Loading dataloader successed")

        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
            
        model.to(args.device)
        logger.info("Loading model successed")

        print(f"now device is {args.device}")
        train_epochs(train_loader, valid_loader, test_loader, model, optimizer, scheduler, args, best_loss)

        
    if(args.mode == 'infer'):
        args.latest_checkpoint = "../ckpts/finetune_ckpts/XX.pth"
        best_loss = None
        latest_checkpoint = args.latest_checkpoint
        # dataset
        logger.info("Loading dataset ......")

        model = MainModel(args)
        print_model_info(model,level=2)
        tokenizer_org = model.tokenizer
        
        test_dataset = Mol_predict_Dataset(split = "test",
                                  tokenizer_org = tokenizer_org,
                                  args = args
                                 )
        logger.info("Loading dataset successed")

        logger.info("Loading dataloader ......")

        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
        logger.info("Loading dataloader successed")

        state_dict = torch.load(latest_checkpoint, map_location='cpu', weights_only=True)["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
            
        model.to(args.device)
        logger.info("Loading model successed")

        print(f"now device is {args.device}")
        id_list, task_list, truth_list, result_list = test_epochs(test_loader, model, args.device)
        output_df = pd.DataFrame({
            'id': id_list,
            'task': task_list,
            'truth': truth_list,
            'result': result_list
        })
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_df.to_csv(f'infer_result/results_{timestamp}.csv', index=False)