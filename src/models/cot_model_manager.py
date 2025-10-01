# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import GenerationConfig

import torch.nn.functional as F
from utils.xutils import print_model_info
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, Qwen2ForCausalLM, AutoConfig
from transformers import T5ForConditionalGeneration

from peft import get_peft_model, LoraConfig, PeftModelForCausalLM, get_peft_config
from copy import deepcopy
import numpy as np


import logging
from torch.nn.functional import cosine_similarity
import re
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # p_t = e^(-CE)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MainModel(nn.Module):
    def __init__(self, config=None):
        super(MainModel, self).__init__()
        self.config = config
        self.task2id = {
            "mol_cap": 0, "smiles_2_iupac": 1, "bbbp": 2, "clintox": 3,
            "mol_gen": 4, "iupac_2_smiles": 5, "forward reaction prediction": 6, "retrosynthesis": 7,
            "esol": 8, "lipo": 9
        }
        self.task2lora = {
            "mol_cap": "lora_0",
            "smiles_2_iupac": "lora_1",
            "bbbp": "lora_2",
            "clintox": "lora_3",
            "mol_gen": "lora_4",
            "iupac_2_smiles": "lora_4",
            "forward reaction prediction": "lora_5",
            "retrosynthesis": "lora_5",
            "esol": "lora_6",
            "lipo": "lora_7"
        }
        self.lora_names = ["lora_0", "lora_1", "lora_2", "lora_3", "lora_4", "lora_5", "lora_6", "lora_7"]
        self.lora_inf_names = ["lora_inf_0", "lora_inf_1", "lora_inf_2", "lora_inf_3", "lora_inf_4", "lora_inf_5", "lora_inf_6", "lora_inf_7"]
        self.classification_tasks = ["bbbp", "clintox"]
        self.regression_tasks = ["esol", "lipo"]
        self.generation_tasks = ["mol_cap", "smiles_2_iupac", "mol_gen", "iupac_2_smiles", 
                                 "forward reaction prediction", "retrosynthesis"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_pretrain)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_pretrain)
        self.model_config = AutoConfig.from_pretrained(self.config.model_pretrain)


        special_tokens = [
            "<IUPAC>", "</IUPAC>","<smiles>", "</smiles>", "<REACTION>", "</REACTION>"
        ]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )

 
        self.model = get_peft_model(self.model, lora_config)


        self.lora_weights = nn.ParameterDict()
        self.initial_weights = {}
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for lora_name in self.lora_names + self.lora_inf_names:
            temp_model = get_peft_model(AutoModelForCausalLM.from_pretrained(self.config.model_pretrain), lora_config)
            for name, param in temp_model.named_parameters():
                if "lora" in name:
                    safe_name = f"{lora_name}_{name.replace('.', '_')}"
                    noise = torch.randn_like(param.data) * 0.01
                    self.lora_weights[safe_name] = nn.Parameter((param.data + noise).to(device), requires_grad=True)
                    self.initial_weights[safe_name] = param.data.clone().to(device)
            del temp_model
            torch.cuda.empty_cache()


        hidden_size = self.model.config.hidden_size
        self.clintox_classification_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 2)
        ).to(device)
        
        self.bbbp_classification_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 2)
        ).to(device)
        
        self.esol_regression_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        ).to(device)

        self.lipo_regression_head = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 1)
        ).to(device)

        self.apply_lora_by_name("lora_0")

        self.global_step = 0

        self.tokenizer.pad_token_id = self.model.config.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.classification_loss_fn = nn.CrossEntropyLoss()
        self.focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        self.esol_loss_fn = nn.HuberLoss(reduction='mean', delta=1.0)
        self.lipo_loss_fn = nn.HuberLoss(reduction='mean', delta=4.0)

    def apply_lora_by_name(self, lora_name):

        for name, param in self.model.named_parameters():
            if "lora" in name:
                safe_name = f"{lora_name}_{name.replace('.', '_')}"
                if safe_name in self.lora_weights:
                    param.data = self.lora_weights[safe_name].data
                    param.requires_grad = lora_name in self.lora_inf_names if lora_name.startswith("lora_inf") else True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False

        for param in self.bbbp_classification_head.parameters():
            param.requires_grad = not lora_name.startswith("lora_inf")
        for param in self.clintox_classification_head.parameters():
            param.requires_grad = not lora_name.startswith("lora_inf")
        for param in self.esol_regression_head.parameters():
            param.requires_grad = not lora_name.startswith("lora_inf")
        for param in self.lipo_regression_head.parameters():
            param.requires_grad = not lora_name.startswith("lora_inf")

    def apply_lora(self, task, stage='predict'):

        lora_name = self.task2lora[task]
        if stage == 'inference':
            lora_name = lora_name.replace("lora_", "lora_inf_")
        self.apply_lora_by_name(lora_name)

    def forward(self, inputs, stage='inference'):
    
        self.global_step += 1
        device = inputs['input']['input_ids'].device
    
        input_ids = inputs['input']['input_ids']
        attention_mask = inputs['input']['attention_mask']
        labels = inputs['labels']
        task = self.config.task[0] if isinstance(self.config.task, list) else self.config.task  
    

        lora_name = self.task2lora[task].replace("lora_", "lora_inf_")
        self.apply_lora_by_name(lora_name)
    
  
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = output.loss if output.loss is not None else torch.tensor(0.0, device=device, requires_grad=True)
        logits = output.logits  # [batch, seq_len, vocab_size]
    
        logger.info(f"Task {task}, LoRA {lora_name}, LM Loss: {loss.item()}")
    
        if not torch.isfinite(loss):
            logger.warning(f"Invalid loss: Task {task}, LoRA {lora_name}, Loss: {loss.item()}")
            loss = torch.tensor(0.0, device=device, requires_grad=True)
    
        return {"loss": loss, "logits": logits}

    def generate(self, inputs, stage='predict', num_return_sequences=1):

        task = inputs['task'][0]  
        task_id = self.task2id[task]
        task_max_new_tokens = {
            0: 1024,  # mol_cap
            1: 512,   # smiles_2_iupac
            2: 10,    # bbbp
            3: 10,    # clintox
            4: 512,   # mol_gen
            5: 512,   # iupac_2_smiles
            6: 1024,  # forward reaction prediction
            7: 1024,  # retrosynthesis
            8: 10,    # esol
            9: 10     # lipo
        }
        max_new_tokens = task_max_new_tokens.get(task_id, 512)

        self.model.eval()
        with torch.no_grad():
            self.apply_lora(task, stage=stage)
            input_ids = inputs['prompt']['input_ids']
            attention_mask = inputs['prompt']['attention_mask']
            device = input_ids.device

            if stage == 'predict' and task in self.classification_tasks:
                results = []
                for _ in range(num_return_sequences):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]
                    last_token_hidden = hidden_states[:, -1, :]
                    if task == "clintox":
                        logits = self.clintox_classification_head(last_token_hidden)
                    else:
                        logits = self.bbbp_classification_head(last_token_hidden)
                    probs = torch.softmax(logits, dim=-1)
                    predictions = torch.argmax(probs, dim=-1)
                    results.extend(["Yes" if pred == 1 else "No" for pred in predictions])
            elif stage == 'predict' and task in self.regression_tasks:
                results = []
                for _ in range(num_return_sequences):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]
                    last_token_hidden = hidden_states[:, -1, :]
                    if task == "esol":
                        predictions = self.esol_regression_head(last_token_hidden).squeeze(-1)
                    else:
                        predictions = self.lipo_regression_head(last_token_hidden).squeeze(-1)
                    results.extend([str(round(pred.item(), 2)) for pred in predictions])
            else:
                max_new_tokens = 1024
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=0.8,
                    repetition_penalty=1.2
                )
                results = self.tokenizer.batch_decode(generation_output, skip_special_tokens=False)

        return results
    def generate_clear(self, inputs, stage='predict', num_return_sequences=1):

        task = inputs['task'][0]  
        task_id = self.task2id[task]
        task_max_new_tokens = {
            0: 1024,  # mol_cap
            1: 512,   # smiles_2_iupac
            2: 10,    # bbbp
            3: 10,    # clintox
            4: 512,   # mol_gen
            5: 512,   # iupac_2_smiles
            6: 1024,  # forward reaction prediction
            7: 1024,  # retrosynthesis
            8: 10,    # esol
            9: 10     # lipo
        }
        max_new_tokens = task_max_new_tokens.get(task_id, 512)
    
        self.model.eval()
        with torch.no_grad():
            self.apply_lora(task, stage=stage)
            input_ids = inputs['prompt']['input_ids']
            attention_mask = inputs['prompt']['attention_mask']
            device = input_ids.device
            prompt_len = input_ids.size(1)  # 提示长度
    
            if stage == 'predict' and task in self.classification_tasks:
                results = []
                for _ in range(num_return_sequences):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]
                    last_token_hidden = hidden_states[:, -1, :]
                    if task == "clintox":
                        logits = self.clintox_classification_head(last_token_hidden)
                    else:
                        logits = self.bbbp_classification_head(last_token_hidden)
                    probs = torch.softmax(logits, dim=-1)
                    predictions = torch.argmax(probs, dim=-1)
                    results.extend(["Yes" if pred == 1 else "No" for pred in predictions])
            elif stage == 'predict' and task in self.regression_tasks:
                results = []
                for _ in range(num_return_sequences):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]
                    last_token_hidden = hidden_states[:, -1, :]
                    if task == "esol":
                        predictions = self.esol_regression_head(last_token_hidden).squeeze(-1)
                    else:
                        predictions = self.lipo_regression_head(last_token_hidden).squeeze(-1)
                    results.extend([str(round(pred.item(), 2)) for pred in predictions])
            else:
                max_new_tokens = 768
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=0.8,
                    repetition_penalty=1.2
                )

                results = []
                for i in range(generation_output.size(0)):
                    generated_tokens = generation_output[i, prompt_len:].tolist()
                    decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                    results.append(decoded)
    
            return results