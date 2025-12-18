import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR

import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

from attention_nn.model import SimpleTransformer
from attention_nn.dataset import get_dataloader 
from attention_nn.utils import get_weights_file_path , latest_weights_file_path , top_k_accuracy
from configurations.data_config import read_config


def train():
    parser = argparse.ArgumentParser(description="Train a simple Transformer model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    # Load configuration
    config_path = args.config
    # Read the configuration file
    config = read_config(config_path)

    # Prepare data loaders and tokenizer
    print('Preparing data loaders...')
    train_dataloader, val_dataloader , tokenizer = get_dataloader(config)
    vocab_size = tokenizer.get_vocab_size()
    print(f'Vocabulary size = {vocab_size}')

    # Set device and initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    device = torch.device(device)

    model = SimpleTransformer(config['d_model'],
                            vocab_size,
                            seq_len=config['seq_len'],
                            dropout=config['dropout'],
                            rank=config['rank']).to(device)

    # Initialize the parameters
    for p in model.parameters():
        if p.dim() > 1:
            # nn.init.normal_(p)
            nn.init.xavier_uniform_(p)
    
    # Initialize embedding layer with E_fixed and freeze gradients
    # model.input_embeddings.embedding.weight.data = torch.from_numpy(config['E_fixed']).float().to(device)
    # model.input_embeddings.embedding.weight.requires_grad = False

  
    pad_id = tokenizer.token_to_id("[PAD]")
    CE_loss = nn.CrossEntropyLoss(ignore_index=pad_id,label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],eps=1e-9)
    scheduler = ExponentialLR(optimizer, gamma=0.97)
    writter = SummaryWriter(log_dir=config['experiment_name'],)  

    # Make sure the weights folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    initial_epoch = 0
    global_step = 0
    preload = config['preload']

    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')


    tot_global_steps = config['num_epochs']*len(train_dataloader)
    print(f'Total number of global steps = {config["num_epochs"]*len(train_dataloader)}')
    k = 2
    nprints = 30
    print_every = max(1,tot_global_steps // nprints)
    fraction = 0.5
    num_embeddings_to_write = int(vocab_size * fraction)
    # embeddings = model.input_embeddings.embedding.weight.data.clone().cpu().numpy()
    # writter.add_embedding(embeddings[:num_embeddings_to_write], metadata=[tokenizer.id_to_token(i) for i in range(num_embeddings_to_write)], tag='initial_embeddings')

    for epoch in range(initial_epoch,config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            input = batch['input'].to(device) # (batch_size, seq_len)
            label = batch['label'].to(device) # (batch_size, seq_len)
            mask = batch['attention_mask'].to(device) # (batch_size, seq_len, seq_len)

            # Run inputs through model
            logits = model(input,mask) # (batch_size, seq_len, vocab_size)

            # Compute loss: CE_loss(predictor , target) where:
            # predictor shape = (N_tot , vocab_size) 'unnormalized prob over vocabulary' 
            # target shape (N_tot) 'categorical ground truth'
            loss = CE_loss(logits.view(-1,vocab_size) , label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})


            # Backpropagate the loss
            loss.backward()

            
            # Log the loss
            if global_step % print_every == 0:
                av_train_acc , std_train_acc , train_acc , train_loss = top_k_accuracy(model,train_dataloader,pad_id,device,k=k,is_test=True)
                av_val_acc , std_val_acc , val_acc , val_loss = top_k_accuracy(model,val_dataloader,pad_id,device,k=k,is_test=True)
                writter.add_scalars(f'Accuracy/Top-{k}_per_position',
                                    {'Train': train_acc, 'Validation': val_acc }, global_step)
                
                writter.add_scalars('CE_Loss',
                                   { 'Train': train_loss, 'Validation': val_loss }, global_step)

                writter.add_scalar('Learning Rate', scheduler.get_last_lr()[0], global_step)

                # Compute the norm of gradients separately for each parameter tensor
                norms = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        norms[name] = grad_norm
                writter.add_scalars('Grad Norms', norms, global_step)
                       
                # Write embeddings to TensorBoard
                embeddings = model.input_embeddings.embedding.weight.data.clone().cpu().numpy()
    
                writter.add_embedding(embeddings[:num_embeddings_to_write], 
                                      metadata=[tokenizer.id_to_token(i) for i in range(num_embeddings_to_write)], 
                                      tag='embeddings' , global_step=global_step)

                E = embeddings - embeddings.mean(axis=0, keepdims=True)
                cov = (E.T @ E) / E.shape[0]
                eigvals = np.linalg.eigvalsh(cov)
                writter.add_histogram('Cov-Emb Eigvals', eigvals, global_step)

                key = model.attention_layer.w1.weight.data.clone().cpu().numpy()
                writter.add_histogram('Keys', key , global_step)
                query = model.attention_layer.w2.weight.data.clone().cpu().numpy()
                writter.add_histogram('Queries', query , global_step)

                writter.flush()
            
            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            

            global_step += 1
        # scheduler.step()
        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

    # embeddings = model.input_embeddings.embedding.weight.data.clone().cpu().numpy()
    # Write a random fraction of the embeddings to TensorBoard
    # writter.add_embedding(embeddings[:num_embeddings_to_write], metadata=[tokenizer.id_to_token(i) for i in range(num_embeddings_to_write)], tag='final_embeddings')
    # writter.add_embedding(embeddings, metadata=[tokenizer.id_to_token(i) for i in range(vocab_size)], tag='final_embeddings')
    
    # writter.add_hparams({
    #     "lr": config["lr"],
    #     "batch_size": config["batch_size"],
    #     "rank": config["rank"],
    #     "d_model": config["d_model"]})

    writter.close()

if __name__ == "__main__":
    train()
