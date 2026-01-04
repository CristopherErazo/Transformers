import tqdm
import torch


def train_epoch(model, iterator, optimizer, device, vocab_size : int, CE_loss , global_step: int):
    """
    Train the model for one epoch.
    Args:
        model: The Transformer model to be trained.
        iterator: DataLoader or tqdm iterator providing the training data.
        optimizer: The optimizer for updating model weights.
        device: The device (CPU/GPU) to run the training on.
        vocab_size: Size of the vocabulary.
        CE_loss: Cross-entropy loss function.   
    """
    is_tqdm = isinstance(iterator, tqdm.tqdm)
    model.train()
    num_iterations = len(iterator)
    count = 0   
    for batch in iterator:
        input = batch['input'].to(device) # (batch_size, seq_len)
        label = batch['label'].to(device) # (batch_size, seq_len)
        mask = batch['attention_mask'].to(device) # (batch_size, seq_len, seq_len)

        # Run inputs through model
        logits = model(input,mask) # (batch_size, seq_len, vocab_size)

        # Compute loss: CE_loss(predictor , target) where:
        # predictor shape = (N_tot , vocab_size) 'unnormalized prob over vocabulary' (logits)
        # target shape (N_tot) 'categorical ground truth'
        loss = CE_loss(logits.view(-1,vocab_size) , label.view(-1))
        
        # Backpropagate the loss
        loss.backward()
        
        # Update the weights
        optimizer.step()
        # If is last batch of epoch compute gradient norms before zeroing them
        if count == num_iterations -1:
            grad_norms = {}
            for name, param in model.named_parameters():
                short_name = name.split('.')[-2]
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    grad_norms[short_name] = grad_norm
                else:
                    grad_norms[short_name] = 0.0
        optimizer.zero_grad(set_to_none=True)

        if is_tqdm : 
            iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

        global_step += 1
        count += 1
    return grad_norms




def top_k_accuracy(model,data_loader,pad_id,device,CE_loss,k : int = 5):
    """ Compute the top-k accuracy of the model on the given data loader and loss.

    Args:
        model: The Transformer model to evaluate.
        data_loader: DataLoader providing the evaluation data.
        pad_id: The token ID used for padding (to be ignored in accuracy computation).
        device: The device (CPU/GPU) to run the evaluation on.
        CE_loss: Cross-entropy loss function.
        k (int): The 'k' in top-k accuracy.
        
    Returns:
        tuple: (accuracy (float), average_loss (float))
    """

    model.eval()
    # If dataloader is a tqdm iterator, disable its progress bar during evaluation
    if isinstance(data_loader, tqdm.tqdm):
        data_loader.disable = True

    with torch.no_grad():
        total = 0
        rigth = 0
        tot_loss = 0.0
        for batch in data_loader:
            input = batch['input'].to(device) # (batch_size, seq_len)
            label = batch['label'].to(device) # (batch_size, seq_len)
            attention_mask = batch['attention_mask'].to(device)  # (1, seq_len)&( seq_len, seq_len)
            
            logits = model(input,attention_mask)  # (batch_size, seq_len, vocab_size)
            # compute CE loss if test
            
            vocab_size = logits.size(-1)
            loss = CE_loss(logits.view(-1, vocab_size), label.view(-1))
            tot_loss += loss.item()
            # Get top-k predictions
            topk_probs, topk_indices = torch.topk(logits, k, dim=-1)  # (batch_size, seq_len, k)
            
            # Expand labels to compare with top-k indices
            expanded_labels = label.unsqueeze(-1).expand_as(topk_indices)  # (batch_size, seq_len, k)
            
            # Check if the true label is in the top-k predictions
            correct = (topk_indices == expanded_labels).any(dim=-1)  # (batch_size, seq_len), boolean
            
            # Create mask to ignore PAD tokens
            pad_mask = (label != pad_id)  # (batch_size, seq_len), boolean
            correct = correct & pad_mask  # (batch_size, seq_len), boolean
            # Count correct predictions excluding PAD tokens
            total += torch.sum(pad_mask).item()
            rigth += torch.sum(correct).item()

        # Total accuracy
        acc = rigth / total
    return  acc , tot_loss / len(data_loader)




def validation_write(model,train_dataloader,val_dataloader,writter,pad_id,device,CE_loss,selected_tokens:dict,k:int,global_step:int,grad_norms:dict):

    # Compute metrics for TensorBoard logging
    
    # Top-k accuracies and losses
    train_acc , train_loss = top_k_accuracy(model,train_dataloader,pad_id,device,CE_loss,k=k)
    val_acc , val_loss = top_k_accuracy(model,val_dataloader,pad_id,device,CE_loss,k=k)


    # Computations with embeddings
    embeddings = model.input_embeddings.embedding.weight.data.clone()
    E = embeddings - embeddings.mean(axis=0, keepdims=True)
    cov = (E.T @ E) / E.shape[0]
    eigvals = torch.linalg.eigvalsh(cov).cpu().numpy()

    # Key and Query weights
    key = model.attention_layer.Wk.weight.data.clone().cpu().numpy()
    query = model.attention_layer.Wq.weight.data.clone().cpu().numpy()

    # Write to TensorBoard

    writter.add_scalars('Accuracy',
                        {'Train': train_acc, 'Validation': val_acc }, global_step)
    
    writter.add_scalars('CE_Loss',
                        { 'Train': train_loss, 'Validation': val_loss }, global_step)

    writter.add_scalars('Grad Norms', grad_norms, global_step)
    
    idx = selected_tokens['indexes']
    labels = selected_tokens['labels']
    writter.add_embedding(embeddings[idx].cpu().numpy(), metadata=labels, tag='embeddings', global_step=global_step)

    writter.add_histogram('E-Cov Eigvals', eigvals, global_step)
    writter.add_histogram('Keys', key , global_step)
    writter.add_histogram('Queries', query , global_step)

    writter.flush()


def train_and_write(model, train_iterator, val_iterator, writter, optimizer, device, vocab_size , CE_loss , global_step, pad_id,selected_tokens,k,print_freq):
    """
    Train the model for one epoch.
    Args:
        model: The Transformer model to be trained.
        iterator: DataLoader or tqdm iterator providing the training data.
        optimizer: The optimizer for updating model weights.
        device: The device (CPU/GPU) to run the training on.
        vocab_size: Size of the vocabulary.
        CE_loss: Cross-entropy loss function.   
    """
    is_tqdm = isinstance(train_iterator, tqdm.tqdm)
    model.train()
    
 
    for batch in train_iterator:
        input = batch['input'].to(device) # (batch_size, seq_len)
        label = batch['label'].to(device) # (batch_size, seq_len)
        mask = batch['attention_mask'].to(device) # (batch_size, seq_len, seq_len)

        # Run inputs through model
        logits = model(input,mask) # (batch_size, seq_len, vocab_size)

        # Compute loss: CE_loss(predictor , target) where:
        # predictor shape = (N_tot , vocab_size) 'unnormalized prob over vocabulary' (logits)
        # target shape (N_tot) 'categorical ground truth'
        loss = CE_loss(logits.view(-1,vocab_size) , label.view(-1))
        
        # Backpropagate the loss
        loss.backward()

        condition_write = (global_step % print_freq == 0)
        if condition_write:
            # Top-k accuracies and losses
            train_acc , train_loss = top_k_accuracy(model,train_iterator,pad_id,device,CE_loss,k=k)
            val_acc , val_loss = top_k_accuracy(model,val_iterator,pad_id,device,CE_loss,k=k)

            # Computations with embeddings
            embeddings = model.input_embeddings.embedding.weight.data.clone()
            E = embeddings - embeddings.mean(axis=0, keepdims=True)
            cov = (E.T @ E) / E.shape[0]
            eigvals = torch.linalg.eigvalsh(cov).cpu().numpy()

            # Key and Query weights
            key = model.attention_layer.Wk.weight.data.clone().cpu().numpy()
            query = model.attention_layer.Wq.weight.data.clone().cpu().numpy()

            # Gradient norms
            grad_norms = {}
            for name, param in model.named_parameters():
                short_name = name.split('.')[-2]
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    grad_norms[short_name] = grad_norm
                else:
                    grad_norms[short_name] = 0.0

            # Write to TensorBoard

            writter.add_scalars('Accuracy',
                                {'Train': train_acc, 'Validation': val_acc }, global_step)

            writter.add_scalars('CE_Loss',
                                { 'Train': train_loss, 'Validation': val_loss }, global_step)

            writter.add_scalars('Grad Norms', grad_norms, global_step)

            idx = selected_tokens['indexes']
            labels = selected_tokens['labels']
            writter.add_embedding(embeddings[idx].cpu().numpy(), metadata=labels, tag='embeddings', global_step=global_step)

            writter.add_histogram('E-Cov Eigvals', eigvals, global_step)
            writter.add_histogram('Keys', key , global_step)
            writter.add_histogram('Queries', query , global_step)

            writter.flush()

        
        # Update the weights
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if is_tqdm : 
            train_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

        global_step += 1




def metrics_computations(model, dataloader, device, CE_loss,sequence_fractions = [0.7, 0.8, 0.9, 1.0]):
    """ Compute the validation loss of the model on the given validation data loader.

    Args:
        model: The Transformer model to evaluate.
        val_dataloader: DataLoader providing the validation data.
        device: The device (CPU/GPU) to run the evaluation on.
        CE_loss: Cross-entropy loss function.
        
    Returns:
        float: average_loss
    """

    model.eval()
    vocab_size = model.vocab_size
    # If dataloader is a tqdm iterator, disable its progress bar during evaluation
    if isinstance(dataloader, tqdm.tqdm):
        dataloader.disable = True

    with torch.no_grad():
        tot_loss = 0.0
        running_entropy = torch.zeros(len(sequence_fractions))
        running_pr = torch.zeros(len(sequence_fractions))
        
        for batch in dataloader:
            input = batch['input'].to(device) # (batch_size, seq_len)
            label = batch['label'].to(device) # (batch_size, seq_len)
            attention_mask = batch['attention_mask'].to(device)  # (1, seq_len)&( seq_len, seq_len)
            tokens_len = batch['tokens_len'].to(device) # (batch_size)
            
            # Forward pass step by step to access inner activations
            x = model.input_embeddings(input)  # (batch_size, seq_len, d_model)
            x = model.positional_encoding(x)  # (batch_size, seq_len, d_model)
            a = model.attention_layer.attention_probabilities(x, attention_mask)  # (batch_size, seq_len, seq_len)
            z = a @ x  # (batch_size, seq_len, d_model)
            x = model.residual_connection(x, z)  # (batch_size, seq_len, d_model)
            logits = torch.matmul(x, model.input_embeddings.embedding.weight.t())  # (batch_size, seq_len, vocab_size)
            
            # Compute CE loss
            loss = CE_loss(logits.view(-1, vocab_size), label.view(-1))
            tot_loss += loss.item()

            # Sparsity measures 
            entropies , prs = sparsity_measure(a, tokens_len, sequence_fractions=sequence_fractions)
            running_entropy += torch.tensor(entropies)
            running_pr += torch.tensor(prs)

        average_loss = tot_loss / len(dataloader)
        
    return tot_loss / len(dataloader) , (running_entropy / len(dataloader)).tolist() , (running_pr / len(dataloader)).tolist()


def sparsity_measure(attention_probs, tokens_len, sequence_fractions = [0.25, 0.5, 0.75, 1.0]):
    """
    Compute the sparcity measure of attention weights.
    Args:
        attention_probs: Attention probabilities tensor of shape (batch_size, seq_len, seq_len).
        tokens_len: Tensor containing the actual lengths of sequences in the batch (batch_size).
    Returns:
        float: Average sparcity measure over the batch.
    """
    batch_size, seq_len, _ = attention_probs.size()


    # Compute last effective token index for each sample in the batch
    last_token_indices = torch.clamp(tokens_len, min=0,max=seq_len)  # (batch_size)

    entropies = []
    prs = []
    for frac in sequence_fractions:
        idx = (last_token_indices * frac).long()

        # For each sample in the batch collect the attention weigths of the last effective token
        attention = attention_probs[torch.arange(batch_size), idx, :]  # (batch_size, seq_len)
        # Compute attention * log(attention) avoiding entries with attention = 0 with a mask
        entropy = torch.nansum(attention*torch.log(attention),dim=1)  # (batch_size)
        # Entropy per sequence length
        entropy = - entropy / torch.log(idx.float()+1)  # (batch_size)
        entropy = entropy.mean()
        # Compute participation ratio
        pr = torch.sum(attention**2,dim=1)  # (batch_size)
        # pr = pr / (idx.float() + 1 ) # (batch_size)
        pr = pr.mean()

        # Save
        entropies.append(entropy.item())
        prs.append(pr.item())


    return entropies , prs


