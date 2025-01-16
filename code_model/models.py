import os

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

from transformers import BertModel, BertTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt


################################################################################################################
#### Base model ################################################################################################
################################################################################################################

class AttentionalDecoder(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(AttentionalDecoder, self).__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.dropout = nn.Dropout(0.1)
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
    def __init__(self, hidden_size, num_heads, dropout=0.0, batch_first=True):
        super(HardThresholdingMultiheadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        # Check if the hidden size is divisible by the number of heads
        if hidden_size % num_heads!= 0:
            raise ValueError("Hidden size must be divisible by the number of heads.")

        # Initialize the query, key, and value projection layers
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)

        # Initialize the dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, average_attn_weights=False):
        """
        Forward pass of the CustomMultiheadAttention.

        Args:
            query (Tensor): The query tensor.
            key (Tensor): The key tensor.
            value (Tensor): The value tensor.
            key_padding_mask (Tensor, optional): The key padding mask. Defaults to None.
            average_attn_weights (bool, optional): Whether to average the attention weights. Defaults to False.

        Returns:
            Tensor: The attention output.
            Tensor: The attention weights.
        """
        # Get the batch size, sequence length, and embedding size
        batch_size, sequence_length, embedding_size = query.size()

        # Check if the batch first is True
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # Project the query, key, and value tensors
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # Reshape the query, key, and value tensors
        query = query.view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_heads, -1).transpose(1, 2)

        # Compute the attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(embedding_size // self.num_heads)

        # Apply the key padding mask
        if key_padding_mask is not None:
            attention_scores = attention_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(1).to(torch.bool),
                float("-inf"),
            )

        # Compute the attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply the dropout
        attention_weights = self.dropout(attention_weights)

        # Compute the attention output
        attention_output = torch.matmul(attention_weights, value)

        # Reshape the attention output
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_size)

        # Compute the average attention weights
        if average_attn_weights:
            attention_weights = attention_weights.mean(dim=1)

        # Return the attention output and weights
        return attention_output, attention_weights

class HardThresholdingAttentionalDecoder(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(HardThresholdingAttentionalDecoder, self).__init__()
        print(hidden_size, num_classes)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, encoder_output, attention_mask):
        query = self.query_proj(encoder_output)
        key = self.key_proj(encoder_output)
        value = self.value_proj(encoder_output)

        key_padding_mask = ~attention_mask.bool()
        
        attn_output, attn_output_weights = self.attention(query=query, key=key, value=value, key_padding_mask=key_padding_mask, average_attn_weights=False)
        # print(attn_output.shape, attn_output_weights.shape) # (N,L,E), (N,L,S)
        cls_attn_weights = attn_output_weights[:, :, 0, :]
        cls_output = attn_output[:, 0, :] # [CLS] token
        
        output = self.fc(cls_output)
        return output, cls_attn_weights

class HardThresholdingBERTClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.1):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        for param in self.bert.parameters(): # freeze BERT encoder
            param.requires_grad = False
        
        self.decoder = AttentionalDecoder(self.bert.config.hidden_size, num_classes) # 768, 3
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



################################################################################################################
#### LoRA fine-tuning on BERT ##################################################################################
################################################################################################################











        