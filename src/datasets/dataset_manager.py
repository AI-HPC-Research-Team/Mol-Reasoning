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



prompt_reasoning_templates = {
    'bbbp': '''
    The above are three possible predictions for BBB penetration.
    <TASK> bbbp </TASK>
    You are a molecular scientist tasked with determining whether the given molecule can penetrate the blood-brain barrier (BBB).
    <smiles> {input_data} </smiles>.
    Please select the most likely answer and provide the reasoning process.
    The reasoning process and answer are enclosed within <think> </think> and <answer> <BOOLEAN> </BOOLEAN> </answer>.
    <think> reasoning process </think>
    <answer> <BOOLEAN> Yes / No </BOOLEAN> </answer>
    ''',
    'clintox': '''
    The above are three possible predictions for clinical toxicity.
    <TASK> clintox </TASK>
    You are a molecular scientist tasked with predicting whether a given molecule is clinically toxic.
    The molecule is represented by SMILES: <smiles> {input_data} </smiles>.
    Please select the most likely answer and provide the reasoning process.
    The reasoning process and answer are enclosed within <think> </think> and <answer> <BOOLEAN> </BOOLEAN> </answer>.
    <think> reasoning process </think>
    <answer> <BOOLEAN> Yes / No </BOOLEAN> </answer>
    ''',
    'mol_cap': '''
    The above are three possible captions.
    <TASK> molecule caption </TASK>
    You are a molecular scientist tasked with generating a caption for the given molecule.
    <smiles> {input_data} </smiles>.
    Please select the most appropriate caption and provide the reasoning process.
    The reasoning process and answer are enclosed within <think> </think> and <answer> </answer>.
    <think> reasoning process </think>
    <answer> molecular description </answer>
    ''',
    'mol_gen': '''
    The above are three possible SMILES strings.
    <TASK> molecule generation </TASK>
    You are a molecular scientist tasked with generating a molecule based on the given description.
    {input_data}.
    Please select the most appropriate SMILES string and provide the reasoning process.
    The reasoning process and answer are enclosed within <think> </think> and <answer> <smiles> </smiles> </answer>.
    <think> reasoning process </think>
    <answer> <smiles> molecular SMILES string </smiles> </answer>
    ''',
    'forward reaction prediction': '''
    The above are three possible reaction products.
    <TASK> forward reaction prediction </TASK>
    You are a chemist tasked with predicting the product of the given chemical reaction.
    <REACTION> {input_data} </REACTION>.
    Please select the most likely product and provide the reasoning process.
    The reasoning process and answer are enclosed within <think> </think> and <answer> <REACTION> </REACTION> </answer>.
    <think> reasoning process </think>
    <answer> <REACTION> forward reaction </REACTION> </answer>
    ''',
    'retrosynthesis': '''
    The above are three possible reactant sets.
    <TASK> retrosynthesis </TASK>
    You are a chemist tasked with predicting the reactants for the given product in a retrosynthetic analysis.
    <REACTION> {input_data} </REACTION>.
    Please select the most likely reactants and provide the reasoning process.
    The reasoning process and answer are enclosed within <think> </think> and <answer> <REACTION> </REACTION> </answer>.
    <think> reasoning process </think>
    <answer> <REACTION> retrosynthesis reaction </REACTION> </answer>
    ''',
    'smiles_2_iupac': '''
    The above are three possible IUPAC names.
    <TASK> SMILES to IUPAC name </TASK>
    You are a chemist tasked with converting the given SMILES string to its IUPAC name.
    <smiles> {input_data} </smiles>.
    Please select the most appropriate IUPAC name and provide the reasoning process.
    The reasoning process and answer are enclosed within <think> </think> and <answer> <IUPAC> </IUPAC> </answer>.
    <think> reasoning process </think>
    <answer> <IUPAC> IUPAC name </IUPAC> </answer>
    ''',
    'iupac_2_smiles': '''
    The above are three possible SMILES strings.
    <TASK> IUPAC name to SMILES </TASK>
    You are a chemist tasked with converting the given IUPAC name to its SMILES string.
    <IUPAC> {input_data} </IUPAC>.
    Please select the most appropriate SMILES string and provide the reasoning process.
    The reasoning process and answer are enclosed within <think> </think> and <answer> <smiles> </smiles> </answer>.
    <think> reasoning process </think>
    <answer> <smiles> molecular SMILES string </smiles> </answer>
    ''',
    'esol': '''
    The above are three possible solubility values.
    <TASK> esol </TASK>
    You are a molecular scientist tasked with predicting the water solubility of the given molecule.
    <smiles> {input_data} </smiles>.
    Please select the most likely solubility value and provide the reasoning process.
    The reasoning process and answer are enclosed within <think> </think> and <answer> <NUMBER> </NUMBER> </answer>.
    <think> reasoning process </think>
    <answer> <NUMBER> numerical value </NUMBER> </answer>
    ''',
    'lipo': '''
    The above are three possible logP values.
    <TASK> lipophilicity </TASK>
    You are a molecular scientist tasked with predicting the lipophilicity (logP) of the given molecule.
    <smiles> {input_data} </smiles>.
    Please select the most likely logP value and provide the reasoning process.
    The reasoning process and answer are enclosed within <think> </think> and <answer> <NUMBER> </NUMBER> </answer>.
    <think> reasoning process </think>
    <answer> <NUMBER> numerical value </NUMBER> </answer>
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
        filtered_data = filtered_data[filtered_data['task'].isin(['forward reaction prediction','retrosynthesis'])]
        #filtered_data = filtered_data[filtered_data['task'].isin(['mol_cap'])]
        #filtered_data = filtered_data[filtered_data['task'].isin(['mol_gen','iupac_2_smiles'])]
        #filtered_data = filtered_data[filtered_data['task'].isin(['mol_gen'])]
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
        self.labels = []
        self.tasks = []
        self.ids = []

        filtered_data = self.data[self.data['split'] == self.split]
        filtered_data = filtered_data[filtered_data['task'].isin(['bbbp', 'clintox', 'esol', 'lipo'])]
        filtered_data = filtered_data.reset_index(drop=True)

        self.data = filtered_data
        self.data_shape = len(filtered_data)

        for _, row in tqdm(filtered_data.iterrows(), total=self.data_shape):
            self.input.append(row["input"])
            self.output.append(row["think"])
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

        # 使用推理训练模板，但不填充option
        prompt_template = prompt_reasoning_templates[self.tasks[i]]
        # 仅填充input_data，option占位符保留为空（将在训练时由LoRA_0填充）
        prompt = prompt_template.format(
            input_data=self.input[i]
        )

        input_str = '<｜User｜>' + prompt + '<｜Assistant｜>'
        # 初始化think和answer为truth（label）
        think_str = f'<think>{self.output[i]}</think>'
        target_str = ''
        if mol_data['task'] in ['bbbp', 'clintox']:
            target_str = '<answer><BOOLEAN>' + str(self.labels[i]) + '</BOOLEAN></answer>'
        elif mol_data['task'] in ['esol', 'lipo']:
            target_str = '<answer><NUMBER>' + str(self.labels[i]) + '</NUMBER></answer>'
        elif mol_data['task'] in ['mol_gen', 'iupac_2_smiles']:
            target_str = '<answer><smiles>' + str(self.labels[i]) + '</smiles></answer>'
        elif mol_data['task'] in ['forward reaction prediction', 'retrosynthesis']:
            target_str = '<answer><REACTION>' + str(self.labels[i]) + '</REACTION></answer>'
        elif mol_data['task'] in ['smiles_2_iupac']:
            target_str = '<answer><IUPAC>' + str(self.labels[i]) + '</IUPAC></answer>'
        elif mol_data['task'] in ['mol_cap']:
            target_str = '<answer>' + str(self.labels[i]) + '</answer>'

        # 编码输入和完整序列
        input_encoded = self.tokenizer_org(
            input_str,
            return_tensors="pt",
            padding=False,
            truncation=True
        )
        full_encoded = self.tokenizer_org(
            input_str + think_str + target_str,
            return_tensors="pt",
            padding=False,
            truncation=True
        )

        # 设置labels，仅计算think和answer部分的损失
        labels = full_encoded['input_ids'].clone()  # [1, seq_len]
        prompt_len = input_encoded['input_ids'].size(1)
        if prompt_len < labels.size(1):
            labels[:, :prompt_len] = -100  # 屏蔽输入部分的损失
        else:
            print(f"Warning: prompt_len {prompt_len} >= input_len {labels.size(1)} for id {self.ids[i]}")

        mol_data['labels'] = labels  # [1, seq_len]
        mol_data['input'] = full_encoded  # 使用完整序列作为输入
        mol_data['prompt'] = input_encoded  # 仅用于验证

        return mol_data

class MolDataset(BaseDataset):
    def __init__(self, split=None, tokenizer_org = None, args = None):
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

        super(MolDataset, self).__init__(config=None)
                                               #for bond in mol.GetBonds())))

    def _load_data(self):
        self.input = []
        self.input_selfies = []
        self.output = []
        self.tasks = []
        self.labels = []
        self.ids = []

        filtered_data = self.data[self.data['split'] == self.split]

        # 更新data_shape为筛选后的数据量
        #self.data = filtered_data
        filtered_data = filtered_data[filtered_data['task'].isin(['esol', 'lipo'])]
        filtered_data = filtered_data.reset_index(drop=True)
        # self.data  = filtered_data
        self.data_shape = len(filtered_data)
        # 使用tqdm包装filtered_data.iterrows()来显示进度条
        for _, row in tqdm(filtered_data.iterrows(), total=self.data_shape):
            self.input.append(row["prompt"])
            self.output.append(row["think"])
            if row['task'] in ['esol', 'lipo']:
                self.labels.append(round(float(row["output"])))
            else:
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
        prompt_template = prompt_templates[self.tasks[i]]
        prompt = prompt_template.format(input_data=self.input[i])
        #input_str = self.input[i] + '<think>'+ str(self.output[i]) + '</think>' + '<answer>'+ str(self.labels[i])+'</answer>' # prompt + output 组合为输入
        #input_str = [{"role": "user", "content": self.input[i]}, {"role": "assistant", "content": '<think>'+ str(self.output[i]) + '</think>' + '<answer>'+ str(self.labels[i])+'</answer>'}]
        input_str = '<｜User｜>'+ prompt+' <｜Assistant｜>' + '<think>'+ str(self.output[i]) + '</think>' + '<answer>'+ str(self.labels[i])+'</answer>'
        input_prompt = '<｜User｜>'+ prompt+' <｜Assistant｜>'
        # 对输入进行 tokenizer 编码
        # 编码
        input_encoded = self.tokenizer_org(
            input_str,
            return_tensors="pt",
            padding=False,
            truncation=True
        )
        prompt_encoded = self.tokenizer_org(
            input_prompt,
            return_tensors="pt",
            padding=False,
            truncation=True
        )
        
        # 设置 labels
        labels = input_encoded['input_ids'].clone()  # [1, seq_len]
        prompt_len = prompt_encoded['input_ids'].size(1)
        # 防止覆盖 <think><answer>
        if prompt_len < labels.size(1):
            labels[:, :prompt_len] = -100
        else:
            print(f"Warning: prompt_len {prompt_len} >= input_len {labels.size(1)} for id {self.ids[i]}")
        mol_data['labels'] = labels  # [1, seq_len]
        
        mol_data['input'] = input_encoded
        mol_data['prompt'] = prompt_encoded
        
        return mol_data