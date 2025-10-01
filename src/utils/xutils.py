import os
import numpy as np
import random
import torch

import datetime

import torch
from torch.nn.utils.rnn import pad_sequence

import logging
logger = logging.getLogger(__name__)
import re
from nltk.translate.bleu_score import sentence_bleu
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

def ToDevice(obj, device):
    if isinstance(obj, (str)):
        return obj
    if isinstance(obj, (int)):
        return obj
    if isinstance(obj, (float)):
        return obj
    if isinstance(obj, dict):
        for k in obj:
            obj[k] = ToDevice(obj[k], device)
        return obj
    elif isinstance(obj, tuple):
        # Convert tuple to list, modify the list, then convert back to tuple
        obj = list(obj)
        for i in range(len(obj)):
            obj[i] = ToDevice(obj[i], device)
        return tuple(obj)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = ToDevice(obj[i], device)
        return obj
    else:
        return obj.to(device)

def print_model_info(model, level=0, prefix=''):
    total_params = 0
    trainable_params = 0

    for name, module in model.named_children():
        total_params_module = sum(p.numel() for p in module.parameters())
        trainable_params_module = sum(p.numel() for p in module.parameters() if p.requires_grad)

        total_params += total_params_module
        trainable_params += trainable_params_module

        print(f"{prefix}Module: {name} | Total parameters: {total_params_module} | Trainable parameters: {trainable_params_module}")

        if level > 0:
            print_model_info(module, level=level-1, prefix=prefix + '  ')

    if prefix == '':
        print(f"Total parameters: {total_params} | Trainable parameters: {trainable_params} | Trainable ratio: {trainable_params / total_params:.2%}")


def custom_collate_fn(batch):
    collated_batch = {}
    elem_keys = batch[0].keys()
    max_length = 1024  # max length

    tasks = [item['task'] for item in batch]
    classification_tasks = ['bbbp', 'clintox']
    regression_tasks = ['esol', 'lipo']
    generation_tasks = ['mol_cap', 'smiles_2_iupac', 'mol_gen', 'iupac_2_smiles', 
                        'forward reaction prediction', 'retrosynthesis']

    for key in elem_keys:
        if key in ['input', 'prompt']:
            input_ids_tensors = [elem[key]['input_ids'].squeeze(0) if elem[key]['input_ids'].dim() > 1 else elem[key]['input_ids'] for elem in batch]
            padded_input_ids = pad_sequence(input_ids_tensors, batch_first=True)
            
            if padded_input_ids.size(1) > max_length:
                padded_input_ids = padded_input_ids[:, :max_length]

            attention_mask_tensors = [elem[key]['attention_mask'].squeeze(0) if elem[key]['attention_mask'].dim() > 1 else elem[key]['attention_mask'] for elem in batch]
            padded_attention_mask = pad_sequence(attention_mask_tensors, batch_first=True)

            if padded_attention_mask.size(1) > max_length:
                padded_attention_mask = padded_attention_mask[:, :max_length]

            collated_batch[key] = {
                'input_ids': padded_input_ids,
                'attention_mask': padded_attention_mask
            }
        elif key in ['truth', 'task', 'id', 'type', 'predict_candidates', 'cot', 'think']:
            collated_batch[key] = [item[key] for item in batch]
        elif key == 'labels':
            labels = [item[key] for item in batch]
            task_types = [task in classification_tasks + regression_tasks for task in tasks]

            if all(task_types):

                labels = [l.squeeze() if l.dim() > 0 else l for l in labels]
                collated_batch[key] = torch.stack(labels)  # [batch]
            elif not any(task_types):  
             
                labels = [l.unsqueeze(0) if l.dim() == 1 else l for l in labels]
                try:
                    max_dim1 = max(label.size(1) for label in labels)
                    padded_labels = [torch.nn.functional.pad(label, (0, max_dim1 - label.size(1)), value=-100) for label in labels]
                    padded_labels = pad_sequence(padded_labels, batch_first=True, padding_value=-100)
                    collated_batch[key] = padded_labels.squeeze(1)  # [batch, seq_len]
                except IndexError:
                    raise ValueError(f"Invalid labels shape in generation tasks: {[l.shape for l in labels]}")
            else:
                raise ValueError("Mixed task types (classification/regression and generation) in batch not supported")
        else:
          
            print(key)
            try:
                padded_data = torch.stack([item[key] for item in batch])
                if padded_data.dim() > 1:
                    padded_data = padded_data.squeeze(1)
                collated_batch[key] = padded_data
            except Exception as e:
                logger.error(f"Error processing key {key}: {e}")
                raise

    return collated_batch

def custom_collate_fn_reason(batch):
    """
    Custom collate function for reason mode, handling sequence labels for generation tasks.
    """
    collated_batch = {}
    elem_keys = batch[0].keys()
    max_length = 2048  # max sequence length

    for key in elem_keys:
        if key in ['input', 'prompt']:
           
            input_ids_tensors = [elem[key]['input_ids'].squeeze(0) if elem[key]['input_ids'].dim() > 1 else elem[key]['input_ids'] for elem in batch]
            padded_input_ids = pad_sequence(input_ids_tensors, batch_first=True, padding_value=0)
            
            if padded_input_ids.size(1) > max_length:
                padded_input_ids = padded_input_ids[:, :max_length]

            attention_mask_tensors = [elem[key]['attention_mask'].squeeze(0) if elem[key]['attention_mask'].dim() > 1 else elem[key]['attention_mask'] for elem in batch]
            padded_attention_mask = pad_sequence(attention_mask_tensors, batch_first=True, padding_value=0)

            if padded_attention_mask.size(1) > max_length:
                padded_attention_mask = padded_attention_mask[:, :max_length]

            collated_batch[key] = {
                'input_ids': padded_input_ids,
                'attention_mask': padded_attention_mask
            }
        elif key in ['graph']:
           
            molecule_batch = Batch.from_data_list([elem[key] for elem in batch])
            collated_batch[key] = molecule_batch
        elif key in ['truth', 'task', 'id', 'type', 'predict_candidates', 'cot', 'think']:
          
            collated_batch[key] = [item[key] for item in batch]
        elif key == 'labels':
           
            labels = [item[key].unsqueeze(0) if item[key].dim() == 1 else item[key] for item in batch]
            try:
                max_dim1 = max(label.size(1) for label in labels)
                padded_labels = [torch.nn.functional.pad(label, (0, max_dim1 - label.size(1)), value=-100) for label in labels]
                padded_labels = pad_sequence(padded_labels, batch_first=True, padding_value=-100)
                collated_batch[key] = padded_labels.squeeze(1)  # [batch, seq_len]
            except IndexError as e:
                logger.error(f"Invalid labels shape: {[l.shape for l in labels]}")
                raise RuntimeError(f"Failed to pad labels: {e}")
        else:
            
            try:
                padded_data = torch.stack([item[key] for item in batch])
                if padded_data.dim() > 1:
                    padded_data = padded_data.squeeze(1)
                collated_batch[key] = padded_data
            except Exception as e:
                logger.error(f"Error processing key {key}: {e}")
                raise

    return collated_batch

def extract_answer(text, task):

    try:
        if task in ['bbbp', 'clintox']:
           
            matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
            if matches:
                last_match = matches[-1].strip().lower()
                if last_match == 'Yes':
                    return 'Yes'
                elif last_match == 'No':
                    return 'No'
                elif last_match == 'yes':
                    return 'Yes'
                elif last_match == 'no':
                    return 'No'
          
            yes_no_matches = re.findall(r'\b(yes|no)\b', text, re.IGNORECASE)
            if yes_no_matches:
                last_match = yes_no_matches[-1].strip().lower()
                if last_match == 'yes':
                    return 'Yes'
                elif last_match == 'no':
                    return 'No'
            return None
        elif task in ['esol', 'lipo']:
           
            matches = re.findall(r'answer>(.*?)</answer', text, re.DOTALL)
            return matches[-1].strip() if matches else None
        elif task in ['mol_gen', 'iupac_2_smiles', 'smiles_2_iupac']:
           
            matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
            return matches[-1].strip() if matches else None
        elif task in ['forward reaction prediction', 'retrosynthesis']:
           
            matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
            return matches[-1].strip() if matches else None
        elif task == 'mol_cap':
            matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
            end_match = re.search(r'^(.*?)<｜end▁of▁sentence｜>', text, re.DOTALL)
            if matches:
                return matches[-1].strip()
            elif end_match:
                return end_match.group(1).strip()
            else:
                return text.strip()
                
    except Exception:
        return None
        
def extract_think(text, task):
 
    try:
        match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        end_match = re.search(r'^(.*?)<｜end▁of▁sentence｜>', text, re.DOTALL)
        if end_match:
            return end_match.group(1).strip()
        return text.strip()
    except Exception:
        return text.strip()


def compute_smiles_similarity(smiles_pred, smiles_true):
    try:
        mol_pred = Chem.MolFromSmiles(smiles_pred)
        mol_true = Chem.MolFromSmiles(smiles_true)
        if mol_pred and mol_true:
            fp_pred = AllChem.GetMorganFingerprintAsBitVect(mol_pred, 2)
            fp_true = AllChem.GetMorganFingerprintAsBitVect(mol_true, 2)
            return DataStructs.TanimotoSimilarity(fp_pred, fp_true)
        return 0.0
    except:
        return 0.0
def tokenize_smiles(smiles):
    return list(smiles)
    
def compute_reward(task, pred_answer, true_answer, think=None, text = None):
    try:
        if task in ['bbbp', 'clintox']:
            y_true = 1 if true_answer.strip().lower() == "yes" else 0
            y_pred = 1 if pred_answer.strip().lower() == "yes" else 0
            answer_reward = 1.0 if y_true == y_pred else 0.0
        elif task in ['esol', 'lipo']:
            try:
                y_true = float(true_answer)
                y_pred = float(pred_answer)
                rmse = ((y_true - y_pred) ** 2) ** 0.5
                answer_reward = 1 / (rmse + 1e-6)
                answer_reward = min(answer_reward, 1.0)
            except:
                answer_reward = -1.0
        elif task in ['mol_cap', 'smiles_2_iupac','forward reaction prediction', 'retrosynthesis']:
            if pred_answer is not None:
                if(true_answer==pred_answer):
                    answer_reward = 1.0
                elif(len(pred_answer)<1):
                    answer_reward = -1.0
                else:
                    reference = [tokenize_smiles(true_answer)]
                    candidate = tokenize_smiles(pred_answer)
                    bleu = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
                    answer_reward = bleu
            else:
                answer_reward = -1.0
        elif task in ['forward reaction prediction','retrosynthesis']:
            if pred_answer is not None:
                try:
                    mol = Chem.MolFromSmiles(pred_answer)
                    if mol:
                        similarity = compute_smiles_similarity(true_answer, pred_answer)
                        score = similarity if mol else 0.5*similarity
                        answer_reward = 1+score
                    else:
                        answer_reward = 0.0
                except:
                    answer_reward = 0.0
            else:
                answer_reward = -1.0
        elif task in ['mol_gen', 'iupac_2_smiles']:
            if pred_answer is not None:
                try:
                    mol = Chem.MolFromSmiles(pred_answer)
                    if mol:
                        similarity = compute_smiles_similarity(true_answer, pred_answer)
                        score = similarity if mol else 0.5*similarity
                        answer_reward = 1+score
                    else:
                        answer_reward = 0.0
                except:
                    answer_reward = 0.0
            else:
                answer_reward = -1.0
        else:
            answer_reward = 0.0


        think_reward = 0.0
        if think:
          
            think_len = len(think.strip())
            length_reward = np.exp(-((think_len - 1569) ** 2) / (2 * 393 ** 2))
            
      
            words = think.strip().split()
            diversity_reward = len(set(words)) / len(words) if words else 0.5
            
    
            think_reward = 0.5 * length_reward + 0.5 * diversity_reward

        reward = 0.8 * answer_reward + 0.2 * think_reward
        return reward
    except Exception as e:
        logger.warning(f"Fail: {e}")
        return 0.0
