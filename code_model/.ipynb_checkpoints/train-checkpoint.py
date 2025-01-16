def train(model, params, experiment_name, add_sparsity_penalty=False):
    '''
    Takes a model and hyperparameters as input, train the model and save everything in a folder with the name 'experiment_name'
    '''
    losses_train = []
    losses_valid = []
    acc_valid = []
    
    best_valid_acc = 0
    p=0
    
    for epoch in range(num_epochs):
    
        model.train()    
        for input_ids, attention_masks, labels in tqdm(dataloader_train):
            torch.cuda.empty_cache()
            
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(input_ids=input_ids, attention_mask=attention_masks)
            if not add_sparsity_penalty:
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels) + sparsity_penalty
                
            loss.backward()
            optimizer.step()
                        
            losses_train.append(loss.item())
    
        model.eval()
        n_true = n_total = 0
        with torch.no_grad():
            for input_ids, attention_masks, labels in tqdm(dataloader_val):
                torch.cuda.empty_cache()
                
                input_ids = input_ids.to(device)
                attention_masks = attention_masks.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs, _ = model(input_ids=input_ids, attention_mask=attention_masks)
                if not add_sparsity_penalty:
                    loss = criterion(outputs, labels)
                else:
                    loss = criterion(outputs, labels) + sparsity_penalty
                
                losses_valid.append(loss.item())
        
                pred = outputs.argmax(dim=1)
                n_true += (pred == labels).sum()
                n_total += len(pred)
                
        acc_valid = n_true/n_total          
    
        current_losses_train = losses_train[-len(dataloader_train):]
        current_losses_valid = losses_valid[-len(dataloader_val):]
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {np.mean(current_losses_train):.4f}, Valid Loss: {np.mean(current_losses_valid):.4f}, Valid Accuracy: {acc_valid:.4f}')
        plt.plot(pd.Series(losses_train))
        plt.plot(pd.Series(losses_train).rolling(10).mean()) # moyenne glissante
        plt.show()
    
        if acc_valid > best_valid_acc:
            best_valid_acc = acc_valid
            print("Validation accuracy improved, saving model...")
            torch.save(model.state_dict(), f'{experiment_name}/best_model.pt')
            p = 0
            print()
        else:
            p += 1
            if p==patience:   
                break

    with open(f'{experiment_name}/infos.txt', 'w') as file:
        file.write()
    print('Done')

