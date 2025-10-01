# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from rdkit import Chem
import os
import csv
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import pandas as pd
import torch.nn as nn
import re
import numpy as np
import json
import random

def valid_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi.strip("\n"))
        #print(mol)
        if mol is not None:
            return True
        else:
            return False
    except:
        return False


class BaseDataset(Dataset): #, ABC):
    def __init__(self, config):
        super(BaseDataset, self).__init__()
        self.config = config
        self._load_data()
        
    @abstractmethod
    def _load_data(self):
        raise NotImplementedError

    def __len__(self):
        return self.data_shape

# 定义范围常量
RANGES = {
    'esol': {'min': -8.8, 'max': 1.6},
    'lipo': {'min': -1.5, 'max': 4.5}
}

# 映射函数：将 output 转换为 0-1000 的类别
def output_to_category(output, task):
    min_val = RANGES[task]['min']
    max_val = RANGES[task]['max']
    # 线性映射到 0-1000
    category = int((output - min_val) / (max_val - min_val) * 1000)
    # 确保类别在 0-1000 范围内
    return max(0, min(1000, category))
    
class Mol_reason_Dataset(BaseDataset):
    def __init__(self, split=None, tokenizer_org=None, args=None):
        data_path = f"{args.dataset_path}/{args.dataset_name}"
        if data_path.endswith('.pkl'):
            self.data = pd.read_pickle(data_path)
        elif data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith('.txt'):
            self.data = pd.read_table(data_path)
        elif data_path.endswith('.json'):
            self.data = pd.read_json(data_path)
        else:
            raise ValueError(f'Unsupported file extension in: {data_path}')

        self.split = split
        self.tokenizer_org = tokenizer_org
        self.task = args.task
        self.args = args

        super(Mol_reason_Dataset, self).__init__(config=None)

    def _load_data(self):
        self.input = []
        self.think = []
        self.labels = []
        self.tasks = []
        self.ids = []
        self.predict_answer_1 = []
        self.predict_answer_2 = []
        self.predict_answer_3 = []
        
        if(self.split == 'all'):
            filtered_data = self.data
        elif(self.split == 'train'):
            filtered_data = self.data
            filtered_data = filtered_data[filtered_data['split'].isin(['train','test'])]
        else:
            filtered_data = self.data[self.data['split'] == self.split]
        filtered_data = filtered_data[filtered_data['task'].isin(self.args.task)]
        filtered_data = filtered_data.reset_index(drop=True)

        self.data = filtered_data
        self.data_shape = len(filtered_data)

        for _, row in tqdm(filtered_data.iterrows(), total=self.data_shape):
            self.input.append(row["input"])
            self.think.append(row["think"])
            if(row["task"] in ["esol","lipo"]):
                category = round(float(row["output"]), 1)
                self.labels.append(category)
            else:
                self.labels.append(row["output"])
            self.ids.append(row["id"])
            self.tasks.append(row["task"])
            self.predict_answer_1.append(row["predict_answer_1"])
            self.predict_answer_2.append(row["predict_answer_2"])
            self.predict_answer_3.append(row["predict_answer_3"])

    def return_df(self):
        return self.data

    def __getitem__(self, i):
        mol_data = {}
        mol_data['id'] = self.ids[i]
        mol_data['task'] = self.tasks[i] 
        mol_data['truth'] = self.labels[i]  # 真实答案，如"Yes"/"No"、数值、SMILES等
        mol_data['think'] = self.think[i]  # <think>...</think>
    
        # 获取候选答案
        predict_answers = [self.predict_answer_1[i], self.predict_answer_2[i], self.predict_answer_3[i]]
        predict_answers = [ans for ans in predict_answers if ans]  # 过滤空值
    
        # # 随机选择1-3个候选答案
        # num_candidates = random.randint(1, min(3, len(predict_answers)))
        selected_candidates = random.sample(predict_answers, 1)
    
        # # 错误注入（10%概率）
        # if random.random() < 0.1:
        #     for j in range(len(selected_candidates)):
        #         if self.tasks[i] in ['bbbp', 'clintox']:
        #             selected_candidates[j] = "No" if selected_candidates[j] == "Yes" else "Yes"
        #         elif self.tasks[i] in ['esol', 'lipo']:
        #             try:
        #                 pred_val = float(selected_candidates[j])
        #                 selected_candidates[j] = str(pred_val + random.uniform(-0.1, 0.1))
        #             except ValueError:
        #                 pass
        #         else:  # 生成任务
        #             selected_candidates[j] = "invalid_" + str(selected_candidates[j])[:10]
    
        # 构造候选答案字符串
        # candidates_str = "\n".join([f"- {ans}" for ans in selected_candidates])
        candidates_str = selected_candidates[0]
        if(self.tasks[i]in ["esol","lipo"]):
            category = round(float(candidates_str), 1)
            candidates_str = category
        
        # 获取prompt模板
        prompt_template = prompt_reasoning_templates[self.tasks[i]]
        if(self.split == 'test'):
            prompt = prompt_template.format(
                input_data=self.input[i],
                candidates=candidates_str
            )
        else:
            prompt = prompt_template.format(
                input_data=self.input[i],
                candidates=self.labels[i]
            )
        if(i==1):
            print(prompt)
        input_str = prompt
    
        # 编码输入
        input_encoded = self.tokenizer_org(
            input_str,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.args.max_length
        )
    
        # 构造目标输出
        think_str = f"<think>{self.think[i]}</think>"
        target_str = ''
        if mol_data['task'] in ['bbbp', 'clintox']:
            #target_str = f"<answer><BOOLEAN>{self.labels[i]}</BOOLEAN></answer>"
            target_str = f"<answer>{self.labels[i]}</answer>"
        elif mol_data['task'] in ['esol', 'lipo']:
            target_str = f"<answer>{self.labels[i]}</answer>"
        elif mol_data['task'] in ['mol_gen', 'iupac_2_smiles']:
            target_str = f"<answer>{self.labels[i]}</answer>"
        elif mol_data['task'] in ['forward reaction prediction', 'retrosynthesis']:
            target_str = f"<answer>{self.labels[i]}</answer>"
        elif mol_data['task'] in ['smiles_2_iupac']:
            target_str = f"<answer>{self.labels[i]}</answer>"
        else:
            target_str = f"<answer>{self.labels[i]}</answer>"
    
        full_encoded = self.tokenizer_org(
            input_str + think_str + target_str,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.args.max_length
        )
    
        # 设置labels，屏蔽输入部分
        labels = full_encoded['input_ids'].clone()  # [1, seq_len]
        prompt_len = input_encoded['input_ids'].size(1)
        if prompt_len < labels.size(1):
            labels[:, :prompt_len] = -100  # 屏蔽输入部分的损失
        else:
            print(f"Warning: prompt_len {prompt_len} >= input_len {labels.size(1)} for id {self.ids[i]}")
    
        mol_data['labels'] = labels  # [1, seq_len]
        mol_data['input'] = full_encoded
        mol_data['prompt'] = input_encoded
        mol_data['cot'] = think_str + target_str
        mol_data['predict_candidates'] = selected_candidates
    
        return mol_data

prompt_reasoning_templates = {
    'bbbp': '''
    You are a molecular scientist tasked with determining whether the given molecule can penetrate the blood-brain barrier (BBB).
    The molecule is <smiles>{input_data}</smiles>.
    To determine penetration, analyze:
    1. Hydrophobicity (logP, ideally 2-5).
    2. Functional groups (e.g., polar groups reduce penetration).
    3. Molecular weight (<500 Da preferred).
    Provide answer in exactly one <answer>...</answer> and output exactly one <think>...</think>.
    <|Assistant|><answer>{candidates}</answer>.
    ''',
    'clintox': '''
    You are a molecular scientist tasked with predicting whether a given molecule is clinically toxic.
    The molecule is <smiles>{input_data}</smiles>.
    To determine toxicity, analyze:
    1. Functional groups (e.g., nitro groups, halogens may increase toxicity).
    2. Hydrophobicity (logP, high values may correlate with toxicity).
    3. Molecular structure (e.g., complex multi-ring structures may be toxic).
    Provide answer in exactly one <answer>...</answer> and output exactly one <think>...</think>.
    <|Assistant|><answer>{candidates}</answer>.
    ''',
    'esol': '''
    You are a molecular scientist tasked with predicting the water solubility of the given molecule: <smiles>{input_data}</smiles>.
    To determine solubility, analyze:
    1. Hydrophobicity (logP, higher values indicate lower solubility).
    2. Functional groups (e.g., polar groups like hydroxyl increase solubility, non-polar groups decrease it).
    3. Molecular weight (higher values may reduce solubility).
    Predict the output as a number from -8.8 to 1.6, and reason in <think>...</think>.
    <|Assistant|><answer>{candidates}</answer>.
    ''',
    'lipo': '''
    You are a molecular scientist tasked with predicting the lipophilicity (logP) of the given molecule: <smiles>{input_data}</smiles>.
    To determine lipophilicity, analyze:
    1. Hydrophobicity (logP, higher values indicate higher lipophilicity).
    2. Functional groups (e.g., non-polar groups like alkyl increase lipophilicity, polar groups like hydroxyl decrease it).
    3. Molecular structure (e.g., aromatic rings or hydrophobic side chains enhance lipophilicity).
    Predict the output as a number from -1.5 to 4.5, and reason in <think>...</think>.
    <|Assistant|><answer>{candidates}</answer>.
    ''',
    'mol_cap': '''
    You are a molecular scientist tasked with generating a caption for the given molecule.
    The molecule is <smiles>{input_data}</smiles>.
    To generate the description, analyze:
    1. Functional groups (e.g., hydroxyl, carboxyl, or aromatic rings, and their chemical properties).
    2. Structural features (e.g., chain, cyclic, or aromatic backbone).
    3. Chemical properties (e.g., hydrophobicity via logP, molecular weight, or polarity).
    Provide answer in exactly one <answer>...</answer> and output exactly one <think>...</think>.
    <|Assistant|><answer>{candidates}</answer>.
    ''',
    'mol_gen': '''
    <|User|> You are a molecular scientist tasked with generating a molecule based on the given description: {input_data}.
    To generate the molecule, analyze:
    1. Functional groups (e.g., hydroxyl, carboxyl, or aromatic rings to match desired properties).
    2. Structural features (e.g., chain, cyclic, or aromatic backbone).
    3. Chemical properties (e.g., hydrophobicity via logP, molecular weight, or polarity).
    Provide answer in exactly one <answer>...</answer> and output exactly one <think>...</think>.
    <|Assistant|><answer>{candidates}</answer>.
    ''',
    'forward reaction prediction': '''
    You are a chemist tasked with predicting the product of the given chemical reaction: {input_data}.
    To predict the product, analyze:
    1. Functional groups (e.g., identify reactive groups like hydroxyl, carbonyl, or halides).
    2. Reaction type (e.g., substitution, addition, elimination, or condensation).
    3. Molecular structure (e.g., steric hindrance, conjugation, or ring formation potential).
    Provide answer in exactly one <answer>...</answer> and output exactly one <think>...</think>.
    <|Assistant|><answer>{candidates}</answer>.
    ''',
    'retrosynthesis': '''
    You are a molecular scientist tasked with predicting the reactants for the retrosynthesis of a given molecule.
    The target molecule is {input_data}.
    To predict the reactants, analyze:
    1. Functional groups (e.g., identify groups like esters, amides, or alkenes that suggest specific reactions).
    2. Reaction type (e.g., disconnection via hydrolysis, reduction, or cycloaddition).
    3. Molecular structure (e.g., potential bond-breaking sites, ring systems, or stereochemistry).
    Provide answer in exactly one <answer>...</answer> and output exactly one <think>...</think>.
    <|Assistant|><answer>{candidates}</answer>.
    ''',
    'smiles_2_iupac': '''
    You are a molecular scientist tasked with converting a given SMILES string into its IUPAC name.
    The molecule is <smiles>{input_data}</smiles>.
    To determine the IUPAC name, analyze:
    1. Functional groups (e.g., hydroxyl, carbonyl, or amine groups to identify key substituents).
    2. Carbon skeleton (e.g., longest carbon chain or ring system for the parent name).
    3. Substituents and stereochemistry (e.g., alkyl groups, double/triple bonds, or chiral centers).
    Provide answer in exactly one <answer>...</answer> and output exactly one <think>...</think>.
    <|Assistant|><answer>{candidates}</answer>.
    ''',
    'iupac_2_smiles': '''
    <|User|> You are a molecular scientist tasked with converting a given IUPAC name into its SMILES string.
    The molecule is <IUPAC>{input_data}</IUPAC>.
    To determine the SMILES string, analyze:
    1. Functional groups (e.g., hydroxyl, carbonyl, or amine indicated by suffixes like -ol, -one, -amine).
    2. Carbon skeleton (e.g., parent chain or ring system defined by the root name).
    3. Substituents and stereochemistry (e.g., alkyl groups, double/triple bonds, or chiral centers).
    Provide answer in exactly one <answer>...</answer> and output exactly one <think>...</think>.
    <|Assistant|><answer>{candidates}</answer>.
    '''
}

prompt_predict_templates = {
    'bbbp': '<TASK> bbbp </TASK> Whether the molecule can penetrate the blood-brain barrier (BBB). <smiles> {input_data} </smiles>',
    'clintox': '<TASK> clintox </TASK> Whether the molecule is clinically toxic or not. <smiles> {input_data} </smiles>',
    'esol': '<TASK> esol </TASK> Predict the water solubility of the molecule. <smiles> {input_data} </smiles>',
    #'lipo': '<TASK> lipophilicity </TASK> Predict the lipophilicity (logP) of the molecule. <smiles> {input_data} </smiles>',
    'lipo': '{input_data}',
    'mol_cap': '''
    <TASK> molecule caption </TASK>
    You are a molecular scientist tasked with generating a caption for the given molecule. 
    <smiles> {input_data} </smiles>
    Please provide a detailed caption describing the molecule in the format <answer> description </answer>.
    ''',
    'mol_gen': '''
    <TASK> molecule generation </TASK>
    You are a molecular scientist tasked with generating a molecule based on the given description. 
    {input_data}
    Please provide the SMILES string of the generated molecule in this format <answer> <smiles> molecular SMILES string </smiles> </answer>.
    ''',
    'forward reaction prediction': '''
    <TASK> forward reaction prediction </TASK>
    You are a chemist tasked with predicting the product of the given chemical reaction. 
    <REACTION> {input_data} </REACTION> 
    Please provide the SMILES string of the product in the format <answer> <REACTION> forward reaction </REACTION> </answer>.
    ''',
    'retrosynthesis': '''
    <TASK> retrosynthesis </TASK>
    You are a chemist tasked with predicting the reactants for the given product in a retrosynthetic analysis. 
    <REACTION> {input_data} </REACTION>  
    Please provide the SMILES string of the reactants in the format <answer> <REACTION> retrosynthesis reaction </REACTION> </answer>.
    ''',
    'smiles_2_iupac': '''
    <TASK> SMILES to IUPAC name </TASK>
    You are a chemist tasked with converting the given SMILES string to its IUPAC name. 
    <smiles> {input_data} </smiles>.  
    Please provide the IUPAC name in the format <answer> <IUPAC> IUPAC name </IUPAC> </answer>.
    ''',
    'iupac_2_smiles': '''
    <TASK> IUPAC name to SMILES </TASK>
    You are a chemist tasked with converting the given IUPAC name to its SMILES string. 
    <IUPAC> {input_data} </IUPAC>.
    Please provide the SMILES string in the format <answer> <smiles> molecular SMILES string </smiles> </answer>.
    '''
}

class Mol_predict_Dataset(BaseDataset):
    def __init__(self, split=None, tokenizer_org = None, args = None, data = None, batch = None):
        if(data is not None):
            self.data = data
        else:
            data_path = f"{args.dataset_path}/{args.dataset_name}"
            if data_path.endswith('.pkl'):
                self.data = pd.read_pickle(data_path)
            elif data_path.endswith('.csv'):
                self.data = pd.read_csv(data_path)
            elif data_path.endswith('.txt'):
                self.data = pd.read_table(data_path)
            elif data_path.endswith('.json'):
                self.data = pd.read_json(data_path)
            else:
                raise ValueError(f'Unsupported file extension in: {data_path}')
            
        self.split = split
        self.tokenizer_org = tokenizer_org
        self.task = args.task
        self.args = args
        self.batch = batch

        super(Mol_predict_Dataset, self).__init__(config=None)
                                               #for bond in mol.GetBonds())))

    def _load_data(self):
        self.input = []
        self.output = []
        self.tasks = []
        self.labels = []
        self.ids = []

        if(self.split == 'all'):
            filtered_data = self.data
        else:
            filtered_data = self.data[self.data['split'] == self.split]
        # if(self.split in ['valid', 'test']):
        #     filtered_data = self.data[self.data['split'] == self.split]
        # else:
        #     filtered_data = self.data[self.data['split'] == self.split]
        #filtered_data = filtered_data[filtered_data['task'].isin(['bbbp', 'clintox', 'esol', 'lipo'])]
        #filtered_data = filtered_data[filtered_data['task'].isin(['clintox'])]
        #filtered_data = filtered_data[filtered_data['task'].isin(['bbbp'])]
        #filtered_data = filtered_data[filtered_data['task'].isin(['lipo'])]
        #filtered_data = filtered_data[filtered_data['task'].isin(['esol'])]
        #filtered_data = filtered_data[filtered_data['task'].isin(['smiles_2_iupac'])]
        #filtered_data = filtered_data[filtered_data['task'].isin(['forward reaction prediction','retrosynthesis'])]
        #filtered_data = filtered_data[filtered_data['task'].isin(['mol_cap'])]
        #filtered_data = filtered_data[filtered_data['task'].isin(['mol_gen','iupac_2_smiles'])]
        filtered_data = filtered_data[filtered_data['task'].isin(['mol_gen'])]
        if(self.batch is not None):
            filtered_data = filtered_data[filtered_data['batch']==self.batch]
        
        filtered_data = filtered_data.reset_index(drop=True)

        self.data  = filtered_data
        self.data_shape = len(filtered_data)
        # 使用tqdm包装filtered_data.iterrows()来显示进度条
        for _, row in tqdm(filtered_data.iterrows(), total=self.data_shape):
            self.input.append(row["input"])
            #self.output.append(row["think"])
            self.labels.append(row["output"])
            self.ids.append(row["id"])
            self.tasks.append(row["task"])
                
    def return_df(self):  
        return self.data
        
    def __getitem__(self, i):
        mol_data = {}
        mol_data['id'] = self.ids[i]
        mol_data['task'] = self.tasks[i]
        mol_data['truth'] = self.labels[i]
        
        # 获取prompt
        prompt_template = prompt_predict_templates[self.tasks[i]]
        prompt = prompt_template.format(input_data=self.input[i])
        
        if self.tasks[i] in ['bbbp', 'clintox']:
            # 分类任务：prompt不加<｜User｜><｜Assistant｜>
            input_str = prompt
            input_encoded = self.tokenizer_org(
                input_str,
                return_tensors="pt",
                padding=False,
                truncation=True
            )
            # labels为1/0
            mol_data['labels'] = torch.tensor(1 if self.labels[i] == "Yes" else 0, dtype=torch.long)
            mol_data['input'] = input_encoded
            mol_data['prompt'] = input_encoded
        elif self.tasks[i] in ['esol', 'lipo']:
            # 回归任务：prompt不加<｜User｜><｜Assistant｜>
            input_str = prompt
            input_encoded = self.tokenizer_org(
                input_str,
                return_tensors="pt",
                padding=False,
                truncation=True
            )
            # labels为浮点数
            mol_data['labels'] = torch.tensor(round(float(self.labels[i]), 2), dtype=torch.float)
            mol_data['input'] = input_encoded
            mol_data['prompt'] = input_encoded
        else:
            # 生成任务：添加<｜User｜><｜Assistant｜>
            input_str = '<｜User｜>' + prompt + '<｜Assistant｜>'
            input_encoded = self.tokenizer_org(
                input_str,
                return_tensors="pt",
                padding=False,
                truncation=True
            )
            # 编码完整序列
            target_str = ''
            if self.tasks[i] in ['mol_gen', 'iupac_2_smiles']:
                target_str = '<answer><smiles>' + str(self.labels[i]) + '</smiles></answer>'
            elif self.tasks[i] in ['forward reaction prediction', 'retrosynthesis']:
                target_str = '<answer><REACTION>' + str(self.labels[i]) + '</REACTION></answer>'
            elif self.tasks[i] in ['smiles_2_iupac']:
                target_str = '<answer><IUPAC>' + str(self.labels[i]) + '</IUPAC></answer>'
            else:  # mol_cap
                target_str = '<answer>' + str(self.labels[i]) + '</answer>'
            
            full_encoded = self.tokenizer_org(
                input_str + target_str,
                return_tensors="pt",
                padding=False,
                truncation=True
            )
            
            # 设置labels，屏蔽输入部分
            labels = full_encoded['input_ids'].clone()  # [1, seq_len]
            prompt_len = input_encoded['input_ids'].size(1)
            if prompt_len < labels.size(1):
                labels[:, :prompt_len] = -100  # 屏蔽输入部分的损失
            else:
                print(f"Warning: prompt_len {prompt_len} >= input_len {labels.size(1)} for id {self.ids[i]}")
            
            mol_data['labels'] = labels
            mol_data['input'] = full_encoded
            mol_data['prompt'] = input_encoded
        
        return mol_data
