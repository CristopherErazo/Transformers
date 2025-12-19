from pathlib import Path
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse

from attention_nn.model import SimpleTransformer
from attention_nn.dataset import get_dataloader 
from attention_nn.train import train_epoch , validation_write , train_and_write

def main():
    parser = argparse.ArgumentParser(description="Train a simple Transformer model.")
    parser.add_argument('--d_model',type=int,default=128, help='Dimension of the model.')
    parser.add_argument('--dataset_size', type=int, default=200, help='Size of the dataset.')
    parser.add_argument('--train_fraction', type=float, default=0.9, help='Fraction of data used for training.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length.')
    parser.add_argument('--dropout', type=float, default=0.01, help='Dropout rate.')
    parser.add_argument('--rank', type=int, default=64, help='Rank of the model.')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--is_tqdm', type=str,default='True', help='Use tqdm for progress bar?')
    parser.add_argument('--datasource', type=str, default='roneneldan/TinyStories', help='Datasource for the dataset.')
    parser.add_argument('--emb_mode', type=str, default='rnd_train', help="Mode of operation: 'rnd_train', 'rnd_fix' , 'given_train', 'given_fix'.")
    parser.add_argument('--frac_embedd', type=float, default=0.1, help='Fraction of embeddings to write.')

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
    emb_mode = args.emb_mode
    frac_embedd = args.frac_embedd
    k = 3  # for top-k accuracy



    print_freq = 100  # Frequency of printing training status
    if is_tqdm:
        print("Using tqdm for progress bars.")

    # Set up logging and model saving paths
    log_dir = f'logs/bsize{batch_size}_lr{lr}/mode_{emb_mode}'
    model_path = f'data/weights/bsize{batch_size}_lr{lr}/mode_{emb_mode}'

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
    print(f'Vocabulary size = {vocab_size}')

    # Select a fraction of embeddings to write during validation
    num_embedd = int(frac_embedd * vocab_size)
    idx = torch.randperm(vocab_size)[:num_embedd]
    labels = [tokenizer.id_to_token(i) for i in idx.tolist()]
    selected_tokens = {'indexes': idx, 'labels': labels}

    # Set device and build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    device = torch.device(device)
    model = SimpleTransformer(d_model, vocab_size, seq_len, dropout, rank).to(device)

    # Initialize the parameters
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)


    if emb_mode == 'rnd_fix':
        print("Freezing embedding layer...")
        model.input_embeddings.embedding.weight.requires_grad = False
    elif emb_mode == 'given_fix' or emb_mode == 'given_train':
        print("Initializing embeddings as last weigths from pretrained model and freezing embedding layer...")
        pretrained_model_folder = f'data/weights/bsize{batch_size}_lr{lr}/mode_rnd_train'
        weights_files = list(Path(pretrained_model_folder).glob('epoch*.pt'))
        if len(weights_files) == 0:
            raise ValueError(f"No weights files found in {pretrained_model_folder}")
        weights_files.sort()
        latest_weights_file = str(weights_files[-1])
        print(f"Loading weights from {latest_weights_file}")
        state = torch.load(latest_weights_file)
        model_state_dict = state['model_state_dict']
        pretrained_embeddings = model_state_dict['input_embeddings.embedding.weight']
        model.input_embeddings.embedding.weight.data = pretrained_embeddings.to(device)
        if emb_mode == 'given_fix':
            print("Freezing embedding layer...")
            model.input_embeddings.embedding.weight.requires_grad = False
        

    pad_id = tokenizer.token_to_id("[PAD]")
    CE_loss = nn.CrossEntropyLoss(ignore_index=pad_id,label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,eps=1e-9)

    
    Path(model_path).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    writter = SummaryWriter(log_dir=log_dir)  

    tot_global_steps = num_epochs*len(train_dataloader)
    print(f'Total number of global steps = {tot_global_steps}')
    k = 2
    nprints = 30
    print_every = max(1,tot_global_steps // nprints)
    global_step = 0
    

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}") if is_tqdm else train_dataloader
        train_and_write(model, batch_iterator, val_dataloader, writter, optimizer, device, vocab_size , CE_loss , global_step, pad_id,selected_tokens,k,print_every)

        # grad_norms = train_epoch(model, batch_iterator, optimizer, device, vocab_size, CE_loss , global_step)
        # validation_write(model,train_dataloader,val_dataloader,writter,pad_id,device,CE_loss,selected_tokens,k,epoch,grad_norms)


        # Save the model at the end of every epoch
        model_filename = f"{model_path}/epoch{epoch:03d}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
       

if __name__ == "__main__":
    main()