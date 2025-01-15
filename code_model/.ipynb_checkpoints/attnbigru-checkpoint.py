"""
This code heavily borrows from a lab of the Altegrad course from Pr. Michalis Vazirgiannis
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import sys
import json
import operator
import numpy as np
import urllib.request
from tqdm import tqdm



class AttentionWithContext(nn.Module):
    """
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    """

    def __init__(self, input_shape, return_coefficients=False, bias=True):
        super(AttentionWithContext, self).__init__()
        self.return_coefficients = return_coefficients

        self.W = nn.Linear(input_shape, input_shape, bias=bias)
        self.tanh = nn.Tanh()
        self.u = nn.Linear(input_shape, 1, bias=False)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.W.weight.data.uniform_(-initrange, initrange)
        self.W.bias.data.uniform_(-initrange, initrange)
        self.u.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        # do not pass the mask to the next layers
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, x, mask=None):
        uit = self.W(x) # compute uit = W . x  where x represents ht
        uit = self.tanh(uit)
        ait = self.u(uit)
        a = torch.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            a = a*mask.double()

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        eps = 1e-9
        a = a / (torch.sum(a, axis=1, keepdim=True) + eps)
        weighted_input = a * x # computes the attentional vector
        if self.return_coefficients:
            return [torch.sum(weighted_input, axis=1), a] ### [attentional vector, coefficients] ###
        else:
            return torch.sum(weighted_input, axis=1) ### attentional vector only ###


path_root = ''
path_to_data = path_root + 'data/'

d = 30 # dimensionality of word embeddings
n_units = 50 # RNN layer dimensionality
drop_rate = 0.5 # dropout
mfw_idx = 2 # index of the most frequent words in the dictionary
padding_idx = 0 # 0 is for the special padding token
oov_idx = 1 # 1 is for the special out-of-vocabulary token
batch_size = 64
nb_epochs = 15
my_patience = 2 # for early stopping strategy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


my_docs_array_train = np.load(path_to_data + 'docs_train.npy')
my_docs_array_test = np.load(path_to_data + 'docs_test.npy')

my_labels_array_train = np.load(path_to_data + 'labels_train.npy')
my_labels_array_test = np.load(path_to_data + 'labels_test.npy')

# load dictionary of word indexes (sorted by decreasing frequency across the corpus)
with open(path_to_data + 'word_to_index.json', 'r') as my_file:
    word_to_index = json.load(my_file)

# invert mapping
index_to_word = {v: k for k, v in word_to_index.items()}
input_size = my_docs_array_train.shape



class Dataset_(Dataset):
    def __init__(self, x, y):
        self.documents = x
        self.labels = y

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        document = self.documents[index]
        label = self.labels[index]
        sample = {
            "document": torch.tensor(document),
            "label": torch.tensor(label),
            }
        return sample


def get_loader(x, y, batch_size=32):
    dataset = Dataset_(x, y)
    data_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            )
    return data_loader




class AttentionBiGRU(nn.Module):
    def __init__(self, input_shape, n_units, index_to_word, dropout=0):
        super(AttentionBiGRU, self).__init__()
        self.embedding = nn.Embedding(len(index_to_word) + 2, # vocab size
                                      d, # dimensionality of embedding space
                                      padding_idx=0)
        self.dropout = nn.Dropout(drop_rate)
        self.gru = nn.GRU(input_size=d,
                          hidden_size=n_units,
                          num_layers=1,
                          bias=True,
                          batch_first=True,
                          bidirectional=True)
        self.attention = AttentionWithContext(n_units * 2,   # the input shape for the attention layer
                                              return_coefficients=True)


    def forward(self, sent_ints):
        sent_wv = self.embedding(sent_ints)
        sent_wv_dr = self.dropout(sent_wv)
        sent_wa, _ = self.gru(sent_wv_dr) # RNN layer
        sent_att_vec, word_att_coeffs = self.attention(sent_wa) # attentional vector for the sent
        sent_att_vec_dr = self.dropout(sent_att_vec)
        return sent_att_vec_dr, word_att_coeffs

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size) (448, 30)
        sent_att_vec_dr, word_att_coeffs = self.module(x_reshape)
        # We have to reshape the output
        if self.batch_first:
            sent_att_vec_dr = sent_att_vec_dr.contiguous().view(x.size(0), -1, sent_att_vec_dr.size(-1))  # (samples, timesteps, output_size)
            word_att_coeffs = word_att_coeffs.contiguous().view(x.size(0), -1, word_att_coeffs.size(-1))  # (samples, timesteps, output_size)
        else:
            sent_att_vec_dr = sent_att_vec_dr.view(-1, x.size(1), sent_att_vec_dr.size(-1))  # (timesteps, samples, output_size)
            word_att_coeffs = word_att_coeffs.view(-1, x.size(1), word_att_coeffs.size(-1))  # (timesteps, samples, output_size)
        return sent_att_vec_dr, word_att_coeffs

class HAN(nn.Module):
    def __init__(self, input_shape, n_units, index_to_word, dropout=0):
        super(HAN, self).__init__()
        self.encoder = AttentionBiGRU(input_shape, n_units, index_to_word, dropout)
        self.timeDistributed = TimeDistributed(self.encoder, True)
        self.dropout = nn.Dropout(drop_rate)
        self.gru = nn.GRU(input_size=n_units * 2,# fill the gap # the input shape of GRU layer
                          hidden_size=n_units,
                          num_layers=1,
                          bias=True,
                          batch_first=True,
                          bidirectional=True)
        self.attention = AttentionWithContext(n_units * 2, # fill the gap # the input shape of between-sentence attention layer
                                              return_coefficients=True)
        self.lin_out = nn.Linear(n_units * 2,   # fill the gap # the input size of the last linear layer
                                 1)
        self.preds = nn.Sigmoid()

    def forward(self, doc_ints):
        sent_att_vecs_dr, word_att_coeffs = self.timeDistributed(doc_ints) # fill the gap # get sentence representation
        doc_sa, _ = self.gru(sent_att_vecs_dr)
        doc_att_vec, sent_att_coeffs = self.attention(doc_sa)
        doc_att_vec_dr = self.dropout(doc_att_vec)
        doc_att_vec_dr = self.lin_out(doc_att_vec_dr)
        return self.preds(doc_att_vec_dr), word_att_coeffs, sent_att_coeffs



def evaluate_accuracy(data_loader, verbose=True):
    model.eval()
    total_loss = 0.0
    ncorrect = ntotal = 0
    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            # inference
            output = model(data["document"].to(device))[0]
            output = output[:, -1] # only last vector
            # total number of examples
            ntotal +=  output.shape[0]
            # number of correct predictions
            predictions = torch.round(output)
            ncorrect += torch.sum(predictions == data["label"].to(device)) #fill me # number of correct prediction - hint: use torch.sum
        acc = ncorrect.item() / ntotal
        if verbose:
          print("validation accuracy: {:3.2f}".format(acc*100))
        return acc



from tqdm import tqdm

model = HAN(input_size, n_units, index_to_word).to(device)
model = model.double()
lr = 0.001  # learning rate
criterion = nn.BCELoss() # fill the gap, use Binary cross entropy from torch.nn: https://pytorch.org/docs/stable/nn.html#loss-functions
optimizer = torch.optim.Adam(model.parameters(), lr=lr) #fill me # use Adam optimizer

def train(x_train=my_docs_array_train,
          y_train=my_labels_array_train,
          x_test=my_docs_array_test,
          y_test=my_labels_array_test,
          word_dict=index_to_word,
          batch_size=batch_size):

    train_data = get_loader(x_train, y_train, batch_size)
    test_data = get_loader(my_docs_array_test, my_labels_array_test, batch_size)

    best_validation_acc = 0.0
    p = 0 # patience

    for epoch in range(1, nb_epochs + 1):
        losses = []
        accuracies = []
        with tqdm(train_data, unit="batch") as tepoch:
            for idx, data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                model.train()
                optimizer.zero_grad()
                input = data['document'].to(device)
                label = data['label'].to(device)
                label = label.double()
                output = model.forward(input)[0]
                output = output[:, -1]
                loss = criterion(output, label) # fill the gap # compute the loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # prevent exploding gradient
                optimizer.step()

                losses.append(loss.item())
                accuracy = torch.sum(torch.round(output) == label).item() / batch_size
                accuracies.append(accuracy)
                tepoch.set_postfix(loss=sum(losses)/len(losses), accuracy=100. * sum(accuracies)/len(accuracies))

        # train_acc = evaluate_accuracy(train_data, False)
        test_acc = evaluate_accuracy(test_data, False)
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}, Validation Accuracy: {:3.2f}%"
              .format(epoch, sum(losses)/len(losses), 100.*test_acc))
        if test_acc >= best_validation_acc:
            best_validation_acc = test_acc
            print("Validation accuracy improved, saving model...")
            torch.save(model.state_dict(), './best_model.pt')
            p = 0
            print()
        else:
            p += 1
            if p==my_patience:
                print("Validation accuracy did not improve for {} epochs, stopping training...".format(my_patience))
    print("Loading best checkpoint...")
    model.load_state_dict(torch.load('./best_model.pt'))
    model.eval()
    print('done.')

train()



















import re
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import sklearn.model_selection as sk
import sklearn.metrics as skm

# Text pre-processing
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Dropout, GlobalAveragePooling1D, Flatten, \
    SpatialDropout1D, Bidirectional
import string
from string import digits
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str("0")

# -----------------------------------------------------------

output_dir = "../lstm_results/"
pd.set_option('display.max_rows', 500)



def get_max_length(df):
    max = 0
    for index, row in df.iterrows():  # format sentence for tokenization
        sentence = row['sentence'].replace(",", "").replace(".", " ") \
            .replace("—", " ").replace("â€", "").replace("  ", " ") \
            .replace(";", "").replace("\n", " ").translate(str.maketrans('', '', string.punctuation))
        words = word_tokenize(sentence)
        if len(words) > max:
            max = len(words)
    return max


def run_lstm(train, test, max_len, seed, epoch_val, b_size):
    train, valid = sk.train_test_split(train, train_size=0.8, random_state=seed)

    X_train = train['sentence'].tolist()
    Y_train = train['label']

    X_test = test['sentence'].tolist()
    Y_test = test['label']

    X_valid = valid['sentence'].tolist()
    Y_valid = valid['label']

    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = '<OOV>'  # out of vocabulary token
    vocab_size = 2000
    tokenizer = Tokenizer(num_words=vocab_size, char_level=False, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_train)
    word_index = tokenizer.word_index
    total_words = len(word_index)

    # Padding
    train_sequences = tokenizer.texts_to_sequences(X_train)
    train_padded = pad_sequences(train_sequences,
                                 maxlen=max_len,
                                 padding=padding_type,
                                 truncating=trunc_type)
    test_sequences = tokenizer.texts_to_sequences(X_test)
    test_padded = pad_sequences(test_sequences,
                                maxlen=max_len,
                                padding=padding_type,
                                truncating=trunc_type)
    valid_sequences = tokenizer.texts_to_sequences(X_valid)
    valid_padded = pad_sequences(valid_sequences,
                                 maxlen=max_len,
                                 padding=padding_type,
                                 truncating=trunc_type)
    print('Shape of train tensor: ', train_padded.shape)
    print('Shape of test tensor: ', test_padded.shape)
    print('Shape of valid tensor: ', valid_padded.shape)

    # Define parameter
    embedding_dim = 16
    batch_size = b_size
    epochs = epoch_val

    # Define Dense Model Architecture
    model = Sequential()
    model.add(Embedding(vocab_size,
                        embedding_dim,
                        input_length=max_len,
                        mask_zero=True))
    model.add(LSTM(4, return_sequences=False)) 
    model.add(Dense(5, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    history = model.fit(train_padded, Y_train, validation_data=(valid_padded, Y_valid), epochs=epochs, shuffle=True,
                        verbose=1, batch_size=batch_size)
    res = model.predict(test_padded)
    res = res.argmax(axis=-1)
    print(res)
    cp = skm.classification_report(Y_test.tolist(), res, output_dict=True)

    val_acc = history.history['val_accuracy'][-1]
    test_acc = cp['weighted avg']['f1-score']

    return val_acc, test_acc


# Hyperparameters
epochs = [10, 20, 30]
batch_sizes = [4, 8, 16, 32]

res_df = {"Dataset": [],
          "Seed": [],
          "Epoch": [],
          "Batch-Size": [],
          "Val-Acc": [],
          "Test-Acc": []}

# Run LSTM for each file and store results for hyperparameter combinations
train_dir = sorted(os.listdir("../training_data/test-and-training/training_data/"))
test_dir = sorted(os.listdir("../training_data/test-and-training/test_data/"))
remove_digits = str.maketrans('', '', digits)

for f in range(len(train_dir)):
    print("Experiment Number: ", f)
    name = train_dir[f].replace(".xlsx", "").replace("-train", "")
    seed = int(re.findall("\d+", name)[0])
    base_name = name.translate(remove_digits)[:-1]
    print(name), print(seed), print(base_name)

    train = pd.read_excel("../training_data/test-and-training/training_data/" + train_dir[f], index_col=False)
    test = pd.read_excel("../training_data/test-and-training/test_data/" + test_dir[f], index_col=False)
    max_len = get_max_length(train)

    for e in epochs:
        for b in batch_sizes:
            val_acc, test_acc = run_lstm(train=train, test=test, max_len=max_len,
                                         seed=seed, epoch_val=e, b_size=b)
            print(val_acc),print(test_acc)
            res_df['Dataset'].append(base_name)
            res_df['Seed'].append(seed)
            res_df['Epoch'].append(e)
            res_df['Batch-Size'].append(b)
            res_df['Val-Acc'].append(val_acc)
            res_df['Test-Acc'].append(test_acc)


print(res_df)
t = pd.DataFrame(res_df)
t.to_excel("../grid_search_results/lstm_results/results_full.xlsx", index=False)
