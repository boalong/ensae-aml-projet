import os

import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt


def train(device, dataloader_train, dataloader_val, model, optimizer, num_epochs, patience, experiment_name, add_sparsity_penalty=False, alpha=10):
    '''
    Takes a model and hyperparameters as input, train the model and save everything in a folder with the name 'experiment_name'
    '''
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    file = open(f'{experiment_name}/training_infos.txt', 'w') # open a file to save the infos
    
    criterion = nn.CrossEntropyLoss()
    
    losses_train = []
    losses_valid = []
    if add_sparsity_penalty:
        sparsity_penalties_train = []
        sparsity_penalties_valid = []
    
    best_loss_valid = np.inf
    p=0
    
    for epoch in range(num_epochs):
    
        model.train()    
        for input_ids, attention_masks, labels in tqdm(dataloader_train):
            torch.cuda.empty_cache()
            
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs, cls_attn_weights = model(input_ids=input_ids, attention_mask=attention_masks)
            if not add_sparsity_penalty:
                loss = criterion(outputs, labels)
            else:
                bce_loss = criterion(outputs, labels)
                sparsity_penalty = alpha / torch.sum( torch.linalg.vector_norm(cls_attn_weights, dim=2) )
                loss = bce_loss + sparsity_penalty
                
            loss.backward()
            optimizer.step()
                        
            losses_train.append(loss.item())
            if add_sparsity_penalty:
                sparsity_penalties_train.append(sparsity_penalty.item())
    
        model.eval()
        trues = []
        preds = []
        with torch.no_grad():
            for input_ids, attention_masks, labels in tqdm(dataloader_val):
                trues.append(labels.cpu().numpy())
                torch.cuda.empty_cache()
                
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs, cls_attn_weights = model(input_ids=input_ids, attention_mask=attention_masks)
                if not add_sparsity_penalty:
                    loss = criterion(outputs, labels)
                else:
                    bce_loss = criterion(outputs, labels)
                    sparsity_penalty = alpha / torch.sum( torch.linalg.vector_norm(cls_attn_weights, dim=2) )
                    loss = bce_loss + sparsity_penalty
                
                losses_valid.append(loss.item())
                if add_sparsity_penalty:
                    sparsity_penalties_valid.append(sparsity_penalty.item())
        
                pred = outputs.argmax(dim=1)
                preds.append(pred.cpu().numpy())

        trues = np.concatenate(trues)
        preds = np.concatenate(preds)                
        acc_valid = np.mean(trues == preds)
        f1_valid = f1_score(trues, preds, average='weighted')
    
        current_losses_train = losses_train[-len(dataloader_train):]
        current_losses_valid = losses_valid[-len(dataloader_val):]
        if add_sparsity_penalty:
            current_sparsity_penalties_train = sparsity_penalties_train[-len(dataloader_train):]
            current_sparsity_penalties_valid = sparsity_penalties_valid[-len(dataloader_val):]

        if not add_sparsity_penalty:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {np.mean(current_losses_train):.4f}, Valid Loss: {np.mean(current_losses_valid):.4f}, Valid Accuracy: {acc_valid:.4f}, Valid F1 score: {f1_valid:.4f}')
            file.write(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {np.mean(current_losses_train):.4f}, Valid Loss: {np.mean(current_losses_valid):.4f}, Valid Accuracy: {acc_valid:.4f}, Valid F1 score: {f1_valid:.4f}\n')
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {np.mean(current_losses_train):.4f}, Valid Loss: {np.mean(current_losses_valid):.4f}, Valid Sparsity penalty: {np.mean(current_sparsity_penalties_valid):.4f}, Valid Accuracy: {acc_valid:.4f}, Valid F1 score: {f1_valid:.4f}')
            file.write(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {np.mean(current_losses_train):.4f}, Valid Loss: {np.mean(current_losses_valid):.4f}, Valid Sparsity penalty: {np.mean(current_sparsity_penalties_valid):.4f}, Valid Accuracy: {acc_valid:.4f}, Valid F1 score: {f1_valid:.4f}\n')
            
        plt.plot(pd.Series(losses_train), label='Train loss')
        plt.plot(pd.Series(losses_train).rolling(10).mean(), label='Train loss, 10-window rolling mean') # moyenne glissante
        plt.legend()
        plt.savefig(f'{experiment_name}/epoch_{str(epoch+1).zfill(3)}.png')
        plt.show()
    
        if np.mean(current_losses_valid) < best_loss_valid:
            best_loss_valid = np.mean(current_losses_valid)
            print("Validation loss improved, saving model...")
            model.save_weights(f'{experiment_name}/best_model.pt')
            p = 0
            print()
        else:
            p += 1
            if p==patience:   
                break
    
    file.close()


def evaluate_on_test(device, dataloader_test, model):
    '''
    Return true labels, predictions, attn weights of the decoder wrt [CLS] token, accuracy and f1 score
    '''
    model.eval()
    
    true = []
    pred = []
    all_cls_attn_weights = []
    for input_ids, attention_masks, labels in tqdm(dataloader_test):
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        with torch.no_grad():
            outputs, cls_attn_weights = model(input_ids=input_ids, attention_mask=attention_masks)
            true.append(labels.cpu().numpy())
            pred.append(outputs.argmax(dim=1).cpu().numpy())
            all_cls_attn_weights.append(cls_attn_weights.cpu().numpy())
    true = np.concatenate(true, axis=0)
    pred = np.concatenate(pred, axis=0)
    cls_attn_weights = np.concatenate(all_cls_attn_weights, axis=0)
    acc_test = np.mean(true == pred)
    f1_test = f1_score(true, pred, average='weighted')

    return true, pred, cls_attn_weights, acc_test, f1_test