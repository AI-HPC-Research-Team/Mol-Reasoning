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
from datasets.cot_dataset_manager import Mol_predict_Dataset, Mol_reason_Dataset

from models.cot_model_manager import MainModel

from torch.optim.lr_scheduler import StepLR
import datetime
from transformers import AutoTokenizer
import time
from copy import deepcopy
from transformers import DataCollatorWithPadding
from utils import *

import os
import trl


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_datasets(split, is_reason=True):
    if is_reason:
        return Mol_reason_Dataset(
            split=split,
            tokenizer_org=tokenizer_org,
            args=args
            )
    else:
        return Mol_predict_Dataset(
            split=split,
            tokenizer_org=tokenizer_org,
            args=args
        )
        
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
            result = model.generate_clear(data, stage = 'inference')
            if(i==1):
                temp_result = result[0]
                print(f"truth:{truth[0]} |Result : {temp_result}")
            i=i+1
            for r in result:
                result_list.append(r)
        
        return id_list, task_list, truth_list, result_list

def val_epochs(loader, model, device, stage='train'):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for mol in loader:
            mol = ToDevice(mol, device)
            outputs = model(mol, stage=stage)
            loss = outputs['loss']
            if torch.isfinite(loss):
                total_loss += loss.item() * mol['input']['input_ids'].size(0)
                num_samples += mol['input']['input_ids'].size(0)
    return total_loss / num_samples if num_samples > 0 else float('inf')

def train_epochs_rl(train_dataset, valid_dataset, model, optimizer, args, best_loss=None):
    reason_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=custom_collate_fn_reason,
                              num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn_reason,
                             num_workers=args.num_workers, pin_memory=True)
    running_loss = AverageMeter()
    step = 0
    loss_values = {"reason_loss": [], "valid_loss": []}
    best_ckpt_path = None
    patience = 0
    device = args.device
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    while patience <= args.patience:
        epoch_loss = []
        reason_loader_iter = tqdm(reason_loader, desc="SFT")
        model.train()
        for mol in reason_loader_iter:
            mol = ToDevice(mol, device)
            outputs = model(mol, stage='inference')
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss.update(loss.detach().cpu().item())
            step += 1
            if step % args.logging_steps == 0:
                logger.info(f"Steps={step} SFT Loss={running_loss.get_average():.4f}")
                epoch_loss.append(running_loss.get_average())
                running_loss.reset()

        loss_values["reason_loss"].append(np.mean(epoch_loss))

        valid_loss = val_epochs(valid_loader, model, device, stage='inference')
        loss_values["valid_loss"].append(valid_loss)

        # 保存最佳模型
        if best_loss is None or valid_loss < best_loss:
            patience = 0
            best_loss = valid_loss
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            if not os.path.exists(args.ckpt_output_path):
                os.makedirs(args.ckpt_output_path)

            if best_ckpt_path and os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)
                print(f"Delete: {best_ckpt_path}")

            ckpt_file = f"sft_reason_{timestamp}.pth"
            best_ckpt_path = os.path.join(args.ckpt_output_path, ckpt_file)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer':optimizer,
                'loss': loss_values
            }, best_ckpt_path)

            message = f"SFT: step={step}, best_loss={best_loss}, valid_loss={valid_loss}, {ckpt_file} "
            print(message)
        else:
            patience += 1
            scheduler.step()

        print(loss_values)
        if patience > args.patience:
            break


    args.patience = 0
    patience = 0
    last_ckpt_file = None
    loss_values = {"reason_loss": [], "valid_loss": []}

    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    args.lr = 1e-5
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    best_loss = None  


    for epoch in range(args.rl_epochs):  
        logger.info(f"========REINFORCE Epoch {epoch + 1}========")
        epoch_loss = []
        reason_loader_iter = tqdm(reason_loader, desc="REINFORCE")
        model.train()
        for mol in reason_loader_iter:
            mol = ToDevice(mol, device)
            input_ids = mol['prompt']['input_ids']
            attention_mask = mol['prompt']['attention_mask']
            tasks = mol['task']
            truths = mol['truth']

            outputs = model.generate_clear(
                {'prompt': {'input_ids': input_ids, 'attention_mask': attention_mask}, 'task': tasks},
                stage='inference',
                num_return_sequences=1
            )


            rewards = []
            for j, output in enumerate(outputs):
                logger.info(f"outputs: {output}")
                pred_answer = extract_answer(output, tasks[j])
                think = extract_think(output, tasks[j])
                logger.info(f"predict: {pred_answer}")
                true_answer = truths[j]  # 直接使用truths[j]
                logger.info(f"truth: {true_answer}")
                task = tasks[j]
                reward = compute_reward(task, pred_answer, true_answer, think, output)
                logger.info(f"reward: {reward}")
                rewards.append(reward)

    
            reward_tensor = torch.tensor(rewards, device=device, dtype=torch.float)
            reward_tensor = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std() + 1e-6)
            outputs = model(mol, stage='inference')
            logits = outputs['logits'] 
            log_probs = torch.log_softmax(logits, dim=-1)
            labels = mol['labels']  
            action_log_probs = []
            for i in range(labels.size(0)):
                seq_log_prob = 0.0
                for t in range(labels.size(1)):
                    if labels[i, t] != -100:
                        seq_log_prob += log_probs[i, t, labels[i, t]]
                action_log_probs.append(seq_log_prob)
            action_log_probs = torch.stack(action_log_probs)  # [batch]
            rl_loss = -torch.mean(action_log_probs * reward_tensor)
            if torch.isfinite(rl_loss):
                rl_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                running_loss.update(rl_loss.detach().cpu().item())
            else:
                logger.warning(f"REINFORCE loss: {rl_loss.item()}")
                running_loss.update(0.0)

            step += 1
            if step % args.logging_steps == 0:
                logger.info(f"Steps={step} REINFORCE Loss={running_loss.get_average():.4f}")
                epoch_loss.append(running_loss.get_average())
                running_loss.reset()


        loss_values["reason_loss"].append(np.mean(epoch_loss))
        valid_loss = val_epochs(valid_loader, model, device, stage='inference')
        loss_values["valid_loss"].append(valid_loss)

        train_dataset = Mol_reason_Dataset(
            split="train",
            tokenizer_org=model.tokenizer,
            args=args
        )
        reason_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, 
                                  collate_fn=custom_collate_fn_reason, 
                                  num_workers=args.num_workers, pin_memory=True)
        
        if best_loss is None or valid_loss < best_loss:
            patience = 0
            best_loss = valid_loss
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            if not os.path.exists(args.ckpt_output_path):
                os.makedirs(args.ckpt_output_path)

            if last_ckpt_file and os.path.exists(last_ckpt_file):
                os.remove(last_ckpt_file)
                print(f"Delete: {last_ckpt_file}")

            ckpt_file = f"rl_reason_{epoch}_{timestamp}.pth"
            last_ckpt_path = os.path.join(args.ckpt_output_path, ckpt_file)
            torch.save({
                'model_state_dict': model.state_dict(),
                'loss': loss_values
            }, last_ckpt_path)

            message = f"REINFORCE: epoch={epoch}, best_loss={best_loss}, valid_loss={valid_loss}, {ckpt_file} "
            print(message)
            last_ckpt_file = last_ckpt_path
            scheduler.step()
        else:
            patience += 1
            
        print(loss_values)
        if patience > args.patience:
            break

    return model, best_loss, optimizer
    
#toy_mol_train_rot_data.csv
def add_arguments(parser):
    """

    """
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task", type=str, default=['mol_gen'])
    parser.add_argument('--dataset_toy', type=bool, default=False)
    parser.add_argument("--dataset_path", type=str, default='../data/cot_data')
    parser.add_argument("--dataset_name", type=str, default='cot_dataset.csv')
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
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--peft", type=str, default='lora')
    parser.add_argument("--stage", type=str, default='inference')
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--rl_epochs", type=int, default=10)


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
        args.latest_checkpoint = "../ckpts/finetune_ckpts/XX.pth"
        best_loss = None
        latest_checkpoint = args.latest_checkpoint
        if latest_checkpoint:
            print(f"Latest checkpoint: {latest_checkpoint}")
        else:
            print("No checkpoint found.")
        
        logger.info("Loading dataset ......")
        model = MainModel(args)
    
        if latest_checkpoint:
            state_dict = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)["model_state_dict"]
            model.load_state_dict(state_dict, strict=True)  
            
        
        print_model_info(model, level=4)
        tokenizer_org = model.tokenizer
    
        train_dataset = load_datasets("train")
        valid_dataset = load_datasets("valid")
        test_dataset = load_datasets("test")
        
        logger.info("Loading dataset succeeded")
    
        logger.info("Loading dataloader ......")
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=custom_collate_fn_reason, 
                                 num_workers=args.num_workers, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn_reason, 
                                 num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn_reason, 
                                num_workers=args.num_workers, pin_memory=True)
        logger.info("Loading dataloader succeeded")
    
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
        
        model.to(args.device)
        logger.info("Loading model succeeded")
        print(f"now device is {args.device}")
        model, best_loss, optimizer = train_epochs_rl(
            train_dataset, valid_dataset, model, optimizer, args, best_loss=None
        )
    if(args.mode == 'infer_predict'):
        args.latest_checkpoint = "../ckpts/finetune_ckpts/best_ckpt.pth"
        best_loss = None
        latest_checkpoint = args.latest_checkpoint
        # dataset
        logger.info("Loading dataset ......")

        model = MainModel(args)
        
        print_model_info(model,level=2)
        tokenizer_org = model.tokenizer
        
        test_dataset = load_datasets("test", is_reason=False)
        
        logger.info("Loading dataset successed")

        logger.info("Loading dataloader ......")

        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn_reason, pin_memory=True)
        logger.info("Loading dataloader successed")

        state_dict = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
        args.device = torch.device(args.device)
        model.apply_lora(args.task[0], stage='predict')
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
        output_df.to_csv(f'infer_result_cot/results_{timestamp}.csv', index=False)
        
    if(args.mode == 'infer_inference'):
        args.latest_checkpoint = "../ckpts/finetune_ckpts/best_ckpt.pth"
        best_loss = None
        latest_checkpoint = args.latest_checkpoint
        # dataset
        logger.info("Loading dataset ......")

        model = MainModel(args)
        
        print_model_info(model,level=2)
        tokenizer_org = model.tokenizer
        
        test_dataset = load_datasets("test")
        
        logger.info("Loading dataset successed")

        logger.info("Loading dataloader ......")

        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, collate_fn=custom_collate_fn_reason, pin_memory=True)
        logger.info("Loading dataloader successed")

        state_dict = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)["model_state_dict"]
        model.load_state_dict(state_dict, strict=True)
    
        args.device = torch.device(args.device)
        model.apply_lora(args.task[0], stage='inference')
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
        output_df.to_csv(f'infer_result_cot/results_{timestamp}.csv', index=False)