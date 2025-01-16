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


# assign seed to numpy and PyTorch
seed=2025
torch.manual_seed(seed)
np.random.seed(seed) 


def load_data():
    '''
    Return train, val and test torch datasets
    '''
    ################################################################################################################
    #### Retrieve data #############################################################################################
    ################################################################################################################
    
    train_dir = sorted([f for f in os.listdir("../training_data/test-and-training/training_data/") if f.endswith('xlsx')])
    test_dir = sorted([f for f in os.listdir("../training_data/test-and-training/test_data/") if f.endswith('xlsx')])
    
    for f in range(len(train_dir[:1])): # on ne s'intéresse que aux fichiers split-combine (le plus général, données de meilleure qualité), au 1er split
        train = pd.read_excel("../training_data/test-and-training/training_data/" + train_dir[f], index_col=False)[['sentence', 'label']]
        test = pd.read_excel("../training_data/test-and-training/test_data/" + test_dir[f], index_col=False)[['sentence', 'label']]
    
    sentences = train['sentence'].tolist()
    labels = train['label'].to_numpy()
    sentences_test = test['sentence'].tolist()
    labels_test = test['label'].to_numpy()


    ################################################################################################################
    #### Tokenization ##############################################################################################
    ################################################################################################################
    
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)
    max_length = 0
    sentence_input = []
    labels_output = []
    for i, sentence in enumerate(sentences):
        if isinstance(sentence, str):
            tokens = tokenizer(sentence)['input_ids']
            sentence_input.append(sentence)
            max_length = max(max_length, len(tokens))
            labels_output.append(labels[i])
        else:
            pass
    max_length=256
    tokens_train = tokenizer(sentence_input, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    labels_train = np.array(labels_output)

    sentence_input_test = []
    labels_output_test = []
    for i, sentence in enumerate(sentences_test):
        if isinstance(sentence, str):
            tokens = tokenizer(sentence)['input_ids']
            sentence_input_test.append(sentence)
            labels_output_test.append(labels_test[i])
        else:
            pass
    tokens_test = tokenizer(sentence_input_test, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    labels_test = np.array(labels_output_test)


    ################################################################################################################
    #### Dataset handling ##########################################################################################
    ################################################################################################################

    input_ids = tokens_train['input_ids']
    attention_masks = tokens_train['attention_mask']
    labels_train = torch.LongTensor(labels_train)
    dataset = TensorDataset(input_ids, attention_masks, labels_train)
    val_length = int(len(dataset) * 0.2)
    train_length = len(dataset) - val_length
    dataset_train, dataset_val = torch.utils.data.random_split(dataset=dataset, lengths=[train_length, val_length]) # create train-val split

    input_ids_test = tokens_test['input_ids']
    attention_masks_test = tokens_test['attention_mask']
    labels_test = torch.LongTensor(labels_test)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)


    return dataset_train, dataset_val, dataset_test    