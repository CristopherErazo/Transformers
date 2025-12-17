from pathlib import Path
import torch
import torch.nn as nn
import math

def get_weights_file_path(config: dict, epoch: str) -> str:
    """ Get the file path for saving/loading model weights for a specific epoch.
     Args:
        config (dict): Configuration dictionary containing model parameters.
            - model_folder (str): The folder name for storing model weights.
            - model_basename (str): The base name for the model weight files.
        epoch (str): The epoch identifier (e.g., 'latest' or a specific epoch number).
    Returns:
        str: The full file path for the model weights.
            Path format: ./logs/<model_folder>/<model_basename><epoch>.pt
    """
    model_folder = config['model_folder']
    model_filename = f"{config['model_filename']}_{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config:dict) -> str|None:
    """ Get the latest model weights file path from the model folder.
    Args:
        config (dict): Configuration dictionary containing model parameters.
            - model_folder (str): The folder name for storing model weights.
            - model_basename (str): The base name for the model weight files.
    Returns:
        str|None: The file path of the latest model weights, or None if no weights found.
    """
    model_folder = config['model_folder']
    model_filename = f"{config['model_filename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

def top_k_accuracy(model,data_loader,pad_id,device,k=5,is_test=False):
    model.eval()

    with torch.no_grad():
        total = 0
        rigth = 0
        tot_loss = 0.0
        success_count = []
        for batch in data_loader:
            input = batch['input'].to(device) # (batch_size, seq_len)
            label = batch['label'].to(device) # (batch_size, seq_len)
            attention_mask = batch['attention_mask'].to(device)  # (1, seq_len)&( seq_len, seq_len)
            
            logits = model(input,attention_mask)  # (batch_size, seq_len, vocab_size)
            # compute CE loss if test
            if is_test:
                vocab_size = logits.size(-1)
                loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
                loss = loss_fn(logits.view(-1, vocab_size), label.view(-1))
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
            success_count.append( correct.int() )  # (batch_size, seq_len), int
            total += torch.sum(pad_mask).item()
            rigth += torch.sum(correct).item()

        
        # Concatenate all success counts
        success_count = torch.cat(success_count, dim=0) # shape (num_batches*batch_size, seq_len)
        av_acc = torch.mean(success_count.float(),dim=0)  # (seq_len)
        st_acc = torch.std(success_count.float(),dim=0)/math.sqrt(success_count.shape[0])  # (seq_len)

        # Total accuracy
        acc = rigth / total
    return av_acc, st_acc, acc , tot_loss / len(data_loader)

            