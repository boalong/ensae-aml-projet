import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertModel

import math
import numpy as np


################################################################################################################
#### Base model ################################################################################################
################################################################################################################

class CustomMultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, threshold=None, temperature=None, dropout=0.1): # batch_first=True
        super(CustomMultiheadAttention, self).__init__()
        self.threshold = threshold
        self.temperature = temperature
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value, key_padding_mask, average_attn_weights=False):
        batch_size, sequence_length, embedding_size = query.size()

        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        query = query.view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(embedding_size // self.num_heads) # [16, 8, 245, 245] (batch_size, num_heads, L, S)
        attention_scores = attention_scores.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(1), # (batch_size, 1, 1, S) # ok
            float("-inf"),
        ) # (batch_size, num_heads, L, S)

        #### Temperature ###############################################################################################
        if self.temperature:
            attention_weights = F.softmax(attention_scores / self.temperature, dim=-1) # (batch_size, num_heads, L, S) ok, on somme sur S la source
        ################################################################################################################
        else:
            attention_weights = F.softmax(attention_scores, dim=-1) # (batch_size, num_heads, L, S) ok, on somme sur S la source

        #### Hard thresholding #########################################################################################
        if self.threshold:
            attention_weights = attention_weights.masked_fill(
                attention_weights < self.threshold,
                0.
            )
            attention_weights = F.normalize(attention_weights, p=1, dim=-1) # renormaliser pour que ça somme à 1
            # print(attention_weights.sum())
        ################################################################################################################

        attention_weights = self.dropout(attention_weights)       
        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_size)

        if average_attn_weights:
            attention_weights = attention_weights.mean(dim=1)

        return attention_output, attention_weights

class AttentionalDecoder(nn.Module):
    def __init__(self, hidden_size, num_classes, threshold=None, temperature=None, dropout=0.1):
        super(AttentionalDecoder, self).__init__()     
        self.attention = CustomMultiheadAttention(hidden_size, num_heads=8, threshold=threshold, temperature=temperature)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, encoder_output, attention_mask):
        key_padding_mask = ~attention_mask.bool()
        attn_output, attn_output_weights = self.attention(query=encoder_output, key=encoder_output, value=encoder_output, key_padding_mask=key_padding_mask, average_attn_weights=False)
        # print(attn_output.shape, attn_output_weights.shape) # (N,L,E), (N,num_heads,L,S)
        cls_attn_weights = attn_output_weights[:, :, 0, :] # [CLS] token

        attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output + encoder_output) # residual connection + layer norm
        cls_output = attn_output[:, 0, :] # [CLS] token
        
        output = self.fc(cls_output)
        return output, cls_attn_weights

class BERTClassifier(nn.Module):
    def __init__(self, num_classes, threshold=None, temperature=None, dropout_rate=0.1):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased') # essayer avec d'autres

        for param in self.bert.parameters(): # freeze BERT encoder
            param.requires_grad = False
        
        self.decoder = AttentionalDecoder(self.bert.config.hidden_size, num_classes, threshold=threshold, temperature=temperature) # (768, 3)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            encoder_output = self.bert(
                input_ids, 
                attention_mask=attention_mask
            ).last_hidden_state
        encoder_output = self.dropout(encoder_output)
        
        output, cls_attn_weights = self.decoder(encoder_output, attention_mask)
        return output, cls_attn_weights

    def save_weights(self, path):
        '''
        Utility function to save the weights of the decoder only
        '''
        state_dict = {
            'decoder': self.decoder.state_dict()
        }
        torch.save(state_dict, path)
    
    def load_weights(self, path):
        '''
        Utility function to load the weights of the decoder only
        '''
        state_dict = torch.load(path)
        self.decoder.load_state_dict(state_dict['decoder'])


################################################################################################################
#### LoRA fine-tuned model #####################################################################################
################################################################################################################

class LoRA(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=32, lora_dropout=0.1):
        super(LoRA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        self.lora_A = nn.Parameter(torch.randn(in_features, r))
        self.lora_B = nn.Parameter(torch.randn(out_features, r))
        self.dropout = nn.Dropout(lora_dropout)

    def forward(self, x):
        A = self.lora_A
        B = self.lora_B
        output = x + self.dropout(torch.matmul(x, A) @ B.T / self.lora_alpha)
        return output

class LoRABERTClassifier(nn.Module):
    def __init__(self, num_classes, threshold=None, temperature=None, dropout_rate=0.1):
        super(LoRABERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters(): # freeze BERT encoder
            param.requires_grad = False
            
        modules_to_replace = []
        for name, module in self.bert.named_modules():
            if isinstance(module, nn.Linear) and ('query' in name or 'key' in name or 'value' in name):
                modules_to_replace.append((name, module))
        
        for name, module in modules_to_replace:
            in_features = module.in_features
            out_features = module.out_features
            lora_module = LoRA(in_features, out_features, r=8, lora_alpha=32, lora_dropout=0.1)
            setattr(self.bert, name, nn.Sequential(module, lora_module))
        
        self.decoder = AttentionalDecoder(self.bert.config.hidden_size, num_classes, threshold=threshold, temperature=temperature) # (768, 3)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask):
        encoder_output = self.bert(
            input_ids, 
            attention_mask=attention_mask
        ).last_hidden_state
        
        encoder_output = self.dropout(encoder_output)
        output = self.decoder(encoder_output, attention_mask)
        return output

    def save_weights(self, path):
        '''
        Utility function to save the weights of the lora layers and the decoder
        '''
        lora_decoder_weights = {
            'lora': {name: param for name, param in self.named_parameters() if 'lora' in name},
            'decoder': {name: param for name, param in self.named_parameters() if 'decoder' in name}
        }
        torch.save(lora_decoder_weights, path)
    
    def load_weights(self, path):
        '''
        Utility function to load the weights of the lora layers and the decoder
        '''
        lora_decoder_weights = torch.load(path)
        lora_state_dict = {k: v for k, v in lora_decoder_weights['lora'].items() if 'lora' in k}
        decoder_state_dict = {k: v for k, v in lora_decoder_weights['decoder'].items() if 'decoder' in k}

        # Load LoRA weights
        lora_params = {name: param for name, param in self.named_parameters() if 'lora' in name}
        lora_params.update(lora_state_dict)
        for name, param in lora_params.items():
            self.state_dict()[name].copy_(param)

        # Load decoder weights
        decoder_params = {name: param for name, param in self.named_parameters() if 'decoder' in name}
        decoder_params.update(decoder_state_dict)
        for name, param in decoder_params.items():
            self.state_dict()[name].copy_(param)        