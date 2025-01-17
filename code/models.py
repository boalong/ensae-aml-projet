import os

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from transformers import BertModel, BertTokenizer, BertForMaskedLM
from peft import get_peft_model, LoraConfig, TaskType

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt


################################################################################################################
#### Base model ################################################################################################
################################################################################################################

class AttentionalDecoder(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout=0.1):
        super(AttentionalDecoder, self).__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, encoder_output, attention_mask):
        query = self.query_proj(encoder_output)
        key = self.key_proj(encoder_output)
        value = self.value_proj(encoder_output)
        key_padding_mask = ~attention_mask.bool()
        attn_output, attn_output_weights = self.attention(query=query, key=key, value=value, key_padding_mask=key_padding_mask, average_attn_weights=False)
        # print(attn_output.shape, attn_output_weights.shape) # (N,L,E), (N,num_heads,L,S)
        cls_attn_weights = attn_output_weights[:, :, 0, :] # [CLS] token

        attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output + encoder_output) # residual connection + layer norm
        cls_output = attn_output[:, 0, :] # [CLS] token
        
        output = self.fc(cls_output)
        return output, cls_attn_weights

class BERTClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.1):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased') # essayer avec d'autres

        for param in self.bert.parameters(): # freeze BERT encoder
            param.requires_grad = False
        
        self.decoder = AttentionalDecoder(self.bert.config.hidden_size, num_classes) # (768, 3)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask, **kwargs):
        with torch.no_grad():
            encoder_output = self.bert(
                input_ids, 
                attention_mask=attention_mask
            ).last_hidden_state
        encoder_output = self.dropout(encoder_output)
        
        output, cls_attn_weights = self.decoder(encoder_output, attention_mask)
        return output, cls_attn_weights

    def save_decoder_weights(self, path):
        '''
        Utility function to save the weights of the decoder only
        '''
        state_dict = {
            'decoder': self.decoder.state_dict()
        }
        torch.save(state_dict, path)
    
    def load_decoder_weights(self, path):
        '''
        Utility function to load the weights of the decoder only
        '''
        state_dict = torch.load(path)
        self.decoder.load_state_dict(state_dict['decoder'])



################################################################################################################
#### Hard thresholding model ###################################################################################
################################################################################################################

class HardThresholdingMultiheadAttention(nn.Module):
    def __init__(self, threshold, hidden_size, num_heads, dropout=0.1): # batch_first=True
        super(HardThresholdingMultiheadAttention, self).__init__()
        self.threshold = threshold
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask, average_attn_weights=False):
        batch_size, sequence_length, embedding_size = query.size()

        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        query = query.view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(embedding_size // self.num_heads)
        attention_scores = attention_scores.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(1).to(torch.bool), # vérfier que pas de pb ici
            float("-inf"),
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        print(attention_weigths.shape)

        #### Hard thresholding #########################################################################################
        attention_weights = attention_weights.masked_fill(
            attention_weights < self.threshold,
            0.
        )
        attention_weights = F.normalize(attention_weights, p=1, dim=-1) # renormaliser pour que ça somme à 1
        ################################################################################################################

        attention_weights = self.dropout(attention_weights) # peut-être que je devrais supprimer le dropout pour avoir des résultats plus stables        
        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_size)

        if average_attn_weights:
            attention_weights = attention_weights.mean(dim=1)

        return attention_output, attention_weights

class HardThresholdingAttentionalDecoder(nn.Module):
    def __init__(self, threshold, hidden_size, num_classes, dropout=0.1):
        super(HardThresholdingAttentionalDecoder, self).__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)        
        self.attention = nn.HardThresholdingMultiheadAttention(threshold=threshold, hidden_size=hidden_size, num_heads=8)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, encoder_output, attention_mask):
        query = self.query_proj(encoder_output)
        key = self.key_proj(encoder_output)
        value = self.value_proj(encoder_output)
        key_padding_mask = ~attention_mask.bool()
        attn_output, attn_output_weights = self.attention(query=query, key=key, value=value, key_padding_mask=key_padding_mask, average_attn_weights=False)
        # print(attn_output.shape, attn_output_weights.shape) # (N,L,E), (N,num_heads,L,S)
        cls_attn_weights = attn_output_weights[:, :, 0, :] # [CLS] token

        attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output + encoder_output) # residual connection + layer norm
        cls_output = attn_output[:, 0, :] # [CLS] token
        
        output = self.fc(cls_output)
        return output, cls_attn_weights

class HardThresholdingBERTClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.1):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        for param in self.bert.parameters(): # freeze BERT encoder
            param.requires_grad = False
        
        self.decoder = HardThresholdingAttentionalDecoder(self.bert.config.hidden_size, num_classes) # 768, 3
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask, **kwargs):
        with torch.no_grad():
            encoder_output = self.bert(
                input_ids, 
                attention_mask=attention_mask
            ).last_hidden_state
        encoder_output = self.dropout(encoder_output)
        
        output, cls_attn_weights = self.decoder(encoder_output, attention_mask)
        return output, cls_attn_weights
        
    def save_decoder_weights(self, path):
        '''
        Utility function to save the weights of the decoder only
        '''
        state_dict = {
            'decoder': self.decoder.state_dict()
        }
        torch.save(state_dict, path)
    
    def load_decoder_weights(self, path):
        '''
        Utility function to load the weights of the decoder only
        '''
        state_dict = torch.load(path)
        self.decoder.load_state_dict(state_dict['decoder'])


################################################################################################################
#### LoRA fine-tuning on BERT ##################################################################################
################################################################################################################

class BERTPretrainer:
    def __init__(self, lora_rank=8, lora_alpha=32, lora_dropout=0.1):
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        peft_config = LoraConfig(
            task_type=TaskType.MASKED_LM,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "key", "value"]
        )
        self.model = get_peft_model(self.model, peft_config)
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def prepare_mlm_input(self, text_batch, mlm_probability=0.15):
        # Tokenize input
        inputs = self.tokenizer(text_batch, padding=True, truncation=True, 
                              max_length=512, return_tensors="pt")
        
        # Create MLM inputs
        labels = inputs.input_ids.clone()
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs.input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        return inputs, labels
    
    def train_step(self, optimizer, text_batch, device):
        self.model.train()
        inputs, labels = self.prepare_mlm_input(text_batch)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)
        
        outputs = self.model(input_ids=inputs.input_ids,
                           attention_mask=inputs.attention_mask,
                           labels=labels)
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        return loss.item()

class LoRABERTClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_lora_path, dropout_rate=0.1):
        super(LoRABERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        peft_config = LoraConfig.from_pretrained(pretrained_lora_path)
        self.bert = get_peft_model(self.bert, peft_config)
        self.bert.load_adapter(pretrained_lora_path)

        for param in self.bert.parameters(): # freeze BERT encoder
            param.requires_grad = False
        
        self.decoder = AttentionalDecoder(self.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, input_ids, attention_mask, **kwargs):
        with torch.no_grad():
            encoder_output = self.bert(
                input_ids, 
                attention_mask=attention_mask
            ).last_hidden_state
        encoder_output = self.dropout(encoder_output)
        
        output, cls_attn_weights = self.decoder(encoder_output, attention_mask)
        return output, cls_attn_weights

    def save_decoder_weights(self, path):
        '''
        Utility function to save the weights of the decoder only
        '''
        state_dict = {
            'decoder': self.decoder.state_dict()
        }
        torch.save(state_dict, path)
    
    def load_decoder_weights(self, path):
        '''
        Utility function to load the weights of the decoder only
        '''
        state_dict = torch.load(path)
        self.decoder.load_state_dict(state_dict['decoder'])



# Example workflow
"""
# First, pretrain with LoRA
texts = ["your", "domain", "specific", "texts"]  # Your unlabeled dataset
lora_path = pretrain_bert_lora(texts)

# Then train the classifier
train_data = ["your", "labeled", "texts"]  # Your labeled dataset
labels = [0, 1, 2]  # Your labels
train_classifier(train_data, labels, lora_path)
"""








        