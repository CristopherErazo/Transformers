from pathlib import Path
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
import time

from attention_nn.model import SimpleTransformer
from attention_nn.dataset import get_dataloader 
from attention_nn.train import top_k_accuracy
from attention_nn.utils import embeddings_computations , get_gradient_norms

def main():
    parser = argparse.ArgumentParser(description="Train a simple Transformer model.")
    parser.add_argument('--d_model',type=int,default=128, help='Dimension of the model.')
    parser.add_argument('--dataset_size', type=int, default=200, help='Size of the dataset.')
    parser.add_argument('--train_fraction', type=float, default=0.9, help='Fraction of data used for training.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length.')
    parser.add_argument('--dropout', type=float, default=0.01, help='Dropout rate.')
    parser.add_argument('--rank', type=int, default=64, help='Rank of the model.')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--is_tqdm', type=str,default='True', help='Use tqdm for progress bar?')
    parser.add_argument('--datasource', type=str, default='roneneldan/TinyStories', help='Datasource for the dataset.')
    parser.add_argument('--frac_embedd', type=float, default=1, help='Fraction of embeddings to write.')

    args = parser.parse_args()

    print("Training configuration:")
    print(args)
    
    d_model = args.d_model
    seq_len = args.seq_len
    dropout = args.dropout
    rank = args.rank
    batch_size = args.batch_size
    train_fraction = args.train_fraction
    dataset_size = args.dataset_size
    datasource = args.datasource
    num_epochs = args.num_epochs
    lr = args.lr
    is_tqdm = args.is_tqdm.lower() == 'true'
    frac_embedd = args.frac_embedd


    # Set up logging and model saving paths
    log_dir = f'logs/dsize{dataset_size}/bsize{batch_size}_lr{lr}'
    model_path = f'data/weights/dsize{dataset_size}/bsize{batch_size}_lr{lr}'
    Path(model_path).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Prepare data loaders and tokenizer
    print('Preparing data loaders...')

    config = {
        'datasource' : datasource,
        'dataset_size' : dataset_size,
        'train_fraction' : train_fraction,
        'seq_len' : seq_len,
        'batch_size' : batch_size}
    
    train_dataloader, val_dataloader , tokenizer = get_dataloader(config)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]")
    print(f'Vocabulary size = {vocab_size}')

    # Set device and build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    device = torch.device(device)
    model = SimpleTransformer(d_model, vocab_size, seq_len, dropout, rank).to(device)

    # Initialize the parameters
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    # Select a fraction of embeddings to write during validation
    num_embedd = int(frac_embedd * vocab_size)
    idx = torch.randperm(vocab_size)[:num_embedd].to(device)
    labels = [tokenizer.id_to_token(i) for i in idx.tolist()]

    
    CE_loss = nn.CrossEntropyLoss(ignore_index=pad_id,label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,eps=1e-9)  
    writter = SummaryWriter(log_dir=log_dir)  


    tot_global_steps = num_epochs*len(train_dataloader)
    nprints = 150
    print_every = max(1,tot_global_steps // nprints)
    global_step = 0
    count_prints = 0
    
    # Run training loop
    time_start = time.time()
    for epoch in range(num_epochs):
        print(f'{epoch = } , running time = {(time.time() - time_start)/60} min')
        torch.cuda.empty_cache()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}") if is_tqdm else train_dataloader
        
        model.train()
        
        for batch in batch_iterator:
            input = batch['input'].to(device) # (batch_size, seq_len)
            label = batch['label'].to(device) # (batch_size, seq_len)
            mask = batch['attention_mask'].to(device) # (batch_size, seq_len, seq_len)

            # Run inputs through model
            logits = model(input,mask) # (batch_size, seq_len, vocab_size)

            # Compute loss
            loss = CE_loss(logits.view(-1,vocab_size) , label.view(-1))
            
            # Backpropagate the loss
            loss.backward()

            condition_write = (global_step % print_every == 0)
            if condition_write:
                # Top-k accuracies and losses
                train_acc , train_loss = top_k_accuracy(model,train_dataloader,pad_id,device,CE_loss,k=3)
                val_acc , val_loss = top_k_accuracy(model,val_dataloader,pad_id,device,CE_loss,k=3)

                # Computations with embeddings
                embeddings = model.input_embeddings.embedding.weight.data.clone()
                eigvals , top_tokens_labels = embeddings_computations(embeddings,tokenizer,m=5)
                
                # Gradient norms
                grad_norms = get_gradient_norms(model)  
                

                # Write to TensorBoard
                writter.add_scalars('Accuracy',
                                    {'Train': train_acc, 'Validation': val_acc }, global_step)

                writter.add_scalars('CE_Loss',
                                    { 'Train': train_loss, 'Validation': val_loss }, global_step)

                writter.add_scalars('Grad Norms', grad_norms, global_step)

                writter.add_histogram('E-Cov Eigvals', eigvals, global_step)

                #?
                # print(global_step, top_tokens_labels[0][0:4])
                writter.add_text('Top Eigenvector Tokens', 
                                 '\n\n'.join([f'EV {i+1}: ' + ', '.join(lb) for i, lb in enumerate(top_tokens_labels)]), 
                                 global_step)
                
                if count_prints in [0, nprints//2 , nprints -1]:
                    writter.add_embedding(embeddings[idx].cpu().numpy(), metadata=labels, tag='embeddings', global_step=global_step)
                count_prints += 1
                writter.flush()

            
            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            #batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            global_step += 1

    

        # Save the model at the end of every epoch
        model_filename = f"{model_path}/epoch{epoch:03d}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
    
    time_end = time.time()
    print(f"Training completed in {(time_end - time_start)/60:.2f} minutes.")
    
    # Save last losses and accuracies with hiperparameters
    # writter.add_hparams(
    #     {'batch_size': batch_size,
    #      'lr': lr},
    #     {'train_accuracy': train_acc,
    #      'val_accuracy': val_acc,
    #      'train_loss': train_loss,
    #      'val_loss': val_loss,
    #      'training_time_min': (time_end - time_start)/60}
    # )
    
    writter.add_graph(model, [input, mask])

    writter.close()
       

if __name__ == "__main__":
    main()
