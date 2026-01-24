import tqdm
import torch
import math
import numpy as np


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




def metrics_computations(model, dataloader, device, CE_loss,data_stats, sorted_rank, sequence_fractions = [0.7, 0.8, 0.9, 1.0]):
    """ Compute the validation loss of the model on the given validation data loader.

    Args:
        model: The Transformer model to evaluate.
        val_dataloader: DataLoader providing the validation data.
        device: The device (CPU/GPU) to run the evaluation on.
        CE_loss: Cross-entropy loss function.
        
    Returns:
        tuple: (average_loss (float), average_entropy (list of float), average_participation_ratio (list of float), attention_patterns (numpy array))
    """

    model.eval()
    vocab_size = model.vocab_size
    num_fractions = len(sequence_fractions)

    fractions_tensor = torch.tensor(sequence_fractions, device=device)  # (num_fractions)
    # If dataloader is a tqdm iterator, disable its progress bar during evaluation
    if isinstance(dataloader, tqdm.tqdm):
        dataloader.disable = True
    P_mu, P_tot, H_mu, H_tot = data_stats
    # P_mu shape (seq_len-1,vocab_size)
    # P_tot shape (vocab_size, )
    # H_mu shape (seq_len-1)
    # H_tot shape ( )

    with torch.no_grad():
        tot_loss = 0.0
        running_entropy = torch.zeros(num_fractions)
        running_pr = torch.zeros(num_fractions)
        running_pos_attended = torch.zeros(num_fractions)
        running_pred_entropy = torch.zeros(model.L)
        running_KL_uniform = 0.0
        running_KL_tot = 0.0
        running_KL_mu = 0.0
        running_average_rank = torch.zeros(num_fractions).to(device)
        
        
        for batch in dataloader:
            input = batch['input'].to(device) # (batch_size, seq_len)
            label = batch['label'].to(device) # (batch_size, seq_len)
            attention_mask = batch['attention_mask'].to(device)  # (1, seq_len)&( seq_len, seq_len)
            tokens_len = batch['tokens_len'].to(device) # (batch_size)
            bs = input.size(0)
            # Forward pass step by step to access inner activations
            e = model.input_embeddings(input)  # (batch_size, seq_len, d_model)
            x = model.positional_encoding(e)  # (batch_size, seq_len, d_model)
            a = model.attention_layer.attention_probabilities(x, attention_mask)  # (batch_size, seq_len, seq_len)
            y = a @ x  # (batch_size, seq_len, d_model)
            z = model.residual_connection(x, y)  # (batch_size, seq_len, d_model)
            if model.U_matrix is not None:  # unmb=True
                U = model.U_matrix
            else:  # unmb=False
                U = model.input_embeddings.embedding.weight.t()
            logits = model.beta*torch.matmul(z, U)/math.sqrt(model.d)  # (batch, seq_len, vocab_size)            probs = torch.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
            probs = torch.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
            # Compute CE loss
            loss = CE_loss(logits.view(-1, vocab_size), label.view(-1))
            tot_loss += loss.item()

            # Sparsity measures 
            entropies , prs , position_attended = sparsity_measure(a, tokens_len, sequence_fractions=sequence_fractions)
            running_entropy += torch.tensor(entropies)
            running_pr += torch.tensor(prs)
            running_pos_attended += torch.tensor(position_attended)

            # Compute entropy of the output distribution per seq position 
            pred_entropy = - torch.nansum(probs * torch.log(probs+1e-12), dim=-1)  # (batch_size, seq_len)
            pred_entropy = pred_entropy.mean(dim=0) # (seq_len)
            running_pred_entropy += pred_entropy.cpu()  # Sum over batches

            # Compare output distribution with data statistics:
            # P_mu shape (seq_len-1,vocab_size) -> per position token relative frequency
            # P_tot shape (vocab_size, ) -> token relative frequency of all dataset
            # Both P_mu and P_tot are torch tensors on the correct device!!

            P_model = probs[:,:-1,:] # # (batch_size, seq_len - 1, vocab_size)
            logP_model = torch.where(P_model > 0, torch.log(P_model), 0)

            # Comparing with uniform distribution
            KL_uniform = - (1/vocab_size) * logP_model.sum(axis=-1).mean()
            running_KL_uniform += KL_uniform.item()

            # Comparing with P_tot (vocab_size, )
            KL_tot = - P_tot[torch.newaxis, torch.newaxis, :] * logP_model  # (batch_size, seq_len - 1, vocab_size)
            KL_tot = KL_tot.sum(axis=-1).mean()  # scalar
            running_KL_tot += KL_tot.item()

            # Comparing with P_mu (seq_len-1,vocab_size)
            KL_mu = - P_mu[torch.newaxis, :, :] * logP_model  # (batch_size, seq_len - 1, vocab_size)
            KL_mu = KL_mu.sum(axis=-1).mean()  # scalar
            running_KL_mu += KL_mu.item()  # Sum over batches

            ##
            # Compute average rank of attended tokens at different sequence fractions
            # Compute indices for each batch and each fraction
            tokens_len_clamped = torch.clamp(tokens_len, max=model.L-1)  # (batch_size)
           
            # Shape: (batch_size, num_fractions)
            idx = (tokens_len_clamped.unsqueeze(1) * fractions_tensor.unsqueeze(0)).long()

            # Prepare batch indices for advanced indexing
            batch_indices = torch.arange(bs, device=a.device).unsqueeze(1).expand(-1, num_fractions)  # (batch_size, num_fractions)

            # Extract attention weights at the required positions
            # attn_weights: (batch_size, num_fractions, seq_len)
            attn_weights = a[batch_indices, idx, :]  # (batch_size, num_fractions, seq_len)

            # Transfor input tensor to ranks with sorter_rank which is a torch tensor of (V, ) already on the correct device
            # basically sorted_rank[i] gives the rank of token with id i
            input_ranks = sorted_rank[input]  # (batch_size, seq_len)

            # Multiply attention weights with input ranks to get weighted ranks
            # attn_ranks: (batch_size, num_fractions, seq_len)
            average_rank = attn_weights * input_ranks.unsqueeze(1)  # (batch_size, num_fractions, seq_len)
            average_rank = average_rank.sum(dim=-1).mean(dim=0) # (num_fractions,)
            

            running_average_rank += average_rank  # Sum over batches

        # Save attention patterns for the maximum token leng
        i_save = torch.argsort(tokens_len, descending=True)[0]
        a_save = a[i_save].cpu().numpy()  # (seq_len, seq_len)  
       

        # Extract also the attention patterns only at the sequence fractions in sequence_fractions
        eff_seq_len = min(int(tokens_len[i_save].item()) , a_save.shape[0])
        idx_of_fractions = [int(eff_seq_len * frac) for frac in sequence_fractions]
        a_in_fractions = a_save[idx_of_fractions , : ]  # (len(sequence_fractions), seq_len)
        
    
    return (tot_loss / len(dataloader),
            (running_entropy / len(dataloader)).tolist() ,
            (running_pr / len(dataloader)).tolist(),
            (running_pos_attended/len(dataloader)).tolist(),
            (running_pred_entropy / len(dataloader)).cpu().numpy(), 
            a_in_fractions,
            (running_KL_uniform / len(dataloader) ) - math.log(vocab_size),  # KL divergence with uniform distribution
            (running_KL_tot / len(dataloader) ) - H_tot.item(),  # KL divergence with P_tot
            (running_KL_mu / len(dataloader) ) - H_mu.mean().item(),  # KL divergence with P_mu
            (running_average_rank / len(dataloader)).tolist()  # average rank of attended tokens at different sequence fractions
            )


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
    position_attended = []  

    for frac in sequence_fractions:
        idx = (last_token_indices * frac).long() # (batch_size)

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

        # Compute the average position attended
        pos_attended = torch.arange(seq_len).to(attention.device).unsqueeze(0) * attention  # (1, seq_len) * (batch_size, seq_len) -> (batch_size, seq_len)
        pos_attended = pos_attended.sum(dim=1)  # (batch_size)
        pos_attended = pos_attended / (idx.float() + 1 ) # (batch_size)
        pos_attended = pos_attended.mean()

        # Save
        entropies.append(entropy.item())
        prs.append(pr.item())
        position_attended.append(pos_attended.item())

    return entropies , prs , position_attended



def get_special_batch(dataloader,L,k=3):

    num_max_len = 0
    while num_max_len < k:
        special_batch = next(iter(dataloader))
        seq_lens = special_batch['tokens_len'].numpy()
        num_max_len = np.sum(seq_lens == L-1)

    # Keep only top k elements in special_batch with the highest token lengths

    lengths = special_batch['tokens_len'].numpy()  # Convert to NumPy array if needed
    top_k_indices = np.argsort(-lengths)[:k]      # Indices of top k lengths

    for key in special_batch.keys():
        value = special_batch[key]
        # If value is a PyTorch tensor, use tensor indexing
        if isinstance(value, torch.Tensor):
            special_batch[key] = value[top_k_indices]
        # If value is a NumPy array, use NumPy indexing
        elif isinstance(value, np.ndarray):
            special_batch[key] = value[top_k_indices]
        # If value is a list, use list comprehension
        elif isinstance(value, list):
            special_batch[key] = [value[i] for i in top_k_indices]
        else:
            # If value is another type, leave it unchanged or handle as needed
            raise TypeError(f"Unsupported data type for key '{key}': {type(value)}")
        
    return special_batch



def model_matrices_diagnostics(model,special_batch,device,sequence_fractions = [0.7, 0.8, 0.9, 1.0],k=15):

    with torch.no_grad():
        input = special_batch['input'].to(device) # (batch_size, seq_len)
        label = special_batch['label'].to(device) # (batch_size, seq_len)
        attention_mask = special_batch['attention_mask'].to(device)  # (1, seq_len)&( seq_len, seq_len)
        tokens_len = special_batch['tokens_len'].to(device) # (batch_size)
        
        # Forward pass step by step to access inner activations
        e = model.input_embeddings(input)  # (batch_size, seq_len, d_model)
        x = model.positional_encoding(e)  # (batch_size, seq_len, d_model)
        a = model.attention_layer.attention_probabilities(x, attention_mask)  # (batch_size, seq_len, seq_len)
        # y = a @ x  # (batch_size, seq_len, d_model)
        # z = model.residual_connection(x, y)  # (batch_size, seq_len, d_model)
        # # logits = model.beta*model.projection(z)  # (batch_size, seq_len, vocab_size)
        # logits = model.beta*torch.matmul(z, model.input_embeddings.embedding.weight.t())/math.sqrt(model.d) # (batch_size, seq_len, vocab_size)
        
        # sequence index for different sequence fractions:
        idx = [int((model.L-1) * frac) for frac in sequence_fractions]
        a_in_fractions = a[:, idx , : ].cpu().numpy()  # (batch_size, len(sequence_fractions), seq_len)

        # For each sample in the batch and each seq frac compute the top k positions attended 
        most_attended_positions = np.argsort(a_in_fractions, axis=-1)[:, :, -k:]  # (batch_size, len(sequence_fractions), k)
        most_attended_positions = most_attended_positions[:, :, ::-1]  # (batch_size, len(sequence_fractions), k) descending order
        tokens_attended = np.take_along_axis(input.cpu().numpy()[:, np.newaxis, :], most_attended_positions, axis=-1)  # (batch_size, len(sequence_fractions), k)

        return (
            a.cpu().numpy(),  # (batch_size, seq_len, seq_len)
            x.cpu().numpy(),  # (batch_size, seq_len, d_model)
            e.cpu().numpy(),  # (batch_size, seq_len, d_model)
            tokens_attended,  # (batch_size, len(sequence_fractions), k)
        )