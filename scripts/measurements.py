import torch
import torch.nn as nn
import argparse
import time
from tqdm import tqdm
import numpy as np

from attention_nn.model import create_model
from attention_nn.dataset import get_dataloader
from attention_nn.train import sparsity_measure , metrics_computations
from attention_nn.utils import embeddings_computations

from configurations.data_config import make_params_dict, save_data

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a simple Transformer model.")
    parser.add_argument('--d_model',type=int,default=256, help='Dimension of the model.')
    parser.add_argument('--dataset_size', type=int, default=1000, help='Size of the dataset.')
    parser.add_argument('--train_fraction', type=float, default=0.9, help='Fraction of data used for training.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--rank', type=int, default=64, help='Rank of the model.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate.')
    parser.add_argument('--is_tqdm', type=str,default='True', help='Use tqdm for progress bar?')
    parser.add_argument('--datasource', type=str, default='roneneldan/TinyStories', help='Datasource for the dataset.')
    parser.add_argument('--freeze_embeddings', type=str, default='False', help='Freeze embeddings during training?')
    parser.add_argument('--freeze_attention', type=str, default='False', help='Freeze attention weights during training?')
    parser.add_argument('--random_positional_encoding', type=str, default='False', help='Use random positional encoding?')
    parser.add_argument('--skip_residual', type=str, default='False', help='Skip residual connections?')
    parser.add_argument('--amplitude', type=float, default=1.0, help='Amplitude of positional encoding.')
    parser.add_argument('--sigma', type=float, default=0.1, help='Standard deviation for parameter initialization.')
    parser.add_argument('--n_prints', type=int, default=20, help='Number of times to print during training.')

    args = parser.parse_args()
    config = vars(args)
    for key in ['is_tqdm', 'freeze_embeddings', 'freeze_attention', 'random_positional_encoding', 'skip_residual']:
        config[key] = True if config[key] == 'True' else False
    
    fix_params = { key : config[key] for key in ['d_model','seq_len','dataset_size']}
    variable_params = { key : config[key] for key in ['batch_size','lr']}

    params = {'fixed' : fix_params,
              'variable': variable_params}
    
    # Rescale learning rate
    config['lr'] = config['lr'] / (config['d_model']**0.5)
    print('Configuration: ', config)

    # Prepare data loaders and tokenizer
    print('Preparing data loaders...')
    train_dataloader, val_dataloader , tokenizer = get_dataloader(config)
    config['vocab_size'] = tokenizer.get_vocab_size()
    print(f'Vocabulary size = {config["vocab_size"]}. Maximum entropy per token = {np.log(config["vocab_size"]):.4f} nats.')
    pad_id = tokenizer.token_to_id("[PAD]")

    # Set device and build model
    model , device = create_model(config)
    print(f'Model built on device: {device}')
   
    CE_loss = nn.CrossEntropyLoss(ignore_index=pad_id,label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],eps=1e-9) 

    sequence_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

    summary = {
        'train_loss': [],
        'val_loss': [],
        'entropy': [],
        'part_ratio': [],
        'sequence_fractions': sequence_fractions,
        'evaluation_steps': [],
        'vocab_size': config['vocab_size'],
        'embedd_eigvals': [],
        'embedd_top_tokens': [],
        'Wk': [],
        'Wq': [],
    }
    data_embeddings = {
        'embeddings': [],
        'embeddings_steps': []
    }

    for name, param in model.named_parameters():
        short_name = name.split('.')[-2]
        short_name = f'grad_{short_name}'
        summary[short_name] = []
        print(f'Parameter: {short_name}, Shape: {param.shape}')

    tot_global_steps = config['num_epochs']*len(train_dataloader)
    nprints = config['n_prints']
    print_every = max(1,tot_global_steps // nprints)
    global_step = 0

    nprints_embeddings = 10
    print_embeddings = max(1,tot_global_steps // nprints_embeddings)
    # Run training loop
    time_start = time.time()
    for epoch in range(config['num_epochs']):
        run_time = (time.time() - time_start)/60 # in minutes
        print(f'epoch {epoch}/{config["num_epochs"]} , running time = {run_time :.5} min')
        
        torch.cuda.empty_cache()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}") if config['is_tqdm'] else train_dataloader
        model.train()
        for batch in batch_iterator:
            input = batch['input'].to(device) # (batch_size, seq_len)
            label = batch['label'].to(device) # (batch_size, seq_len)
            mask = batch['attention_mask'].to(device) # (batch_size, seq_len, seq_len)

            # Run inputs through model
            logits = model(input,mask) # (batch_size, seq_len, vocab_size)
            # Compute loss
            loss = CE_loss(logits.view(-1,config['vocab_size']) , label.view(-1))
            
            # Backpropagate the loss
            loss.backward()
            optimizer.step()
            
            if config['is_tqdm'] : batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Measure validation loss and sparsity metrics
            measure_condition = (global_step % print_every == 0)
            if measure_condition:
                # train_loss , train_entropy , train_pr = metrics_computations(model, train_dataloader, device, CE_loss,sequence_fractions = sequence_fractions)
                val_loss , val_entropy , val_pr = metrics_computations(model, val_dataloader, device, CE_loss,sequence_fractions = sequence_fractions)
                summary['train_loss'].append(loss.item())
                summary['val_loss'].append(val_loss)
                summary['entropy'].append(val_entropy)
                summary['part_ratio'].append(val_pr)
                summary['evaluation_steps'].append(global_step)

                # Record gradient norms
                for name, param in model.named_parameters():
                    short_name = name.split('.')[-2]
                    short_name = f'grad_{short_name}'
                    if param.grad is not None:
                        summary[short_name].append(param.grad.data.norm(2).item())
                    else:
                        summary[short_name].append(0.0)

                # Computation with embeddings
                embeddings = model.input_embeddings.embedding.weight.data.clone()
                eigvals , top_tokens = embeddings_computations(embeddings,tokenizer,m=15,n_tokes=30)
                summary['embedd_eigvals'].append(eigvals)
                summary['embedd_top_tokens'].append(top_tokens)
                # Save attention weights Wk and Wq

                summary['Wk'].append(model.attention_layer.Wk.weight.data.clone().cpu().numpy())
                summary['Wq'].append(model.attention_layer.Wq.weight.data.clone().cpu().numpy())

            # Measure embeddings
            measure_embeddings = (global_step % print_embeddings == 0)
            if measure_embeddings:
                embeddings = model.input_embeddings.embedding.weight.data.clone()
                data_embeddings['embeddings'].append(embeddings.cpu().numpy())
                data_embeddings['embeddings_steps'].append(global_step)
            global_step += 1
            optimizer.zero_grad()
    
    data_embeddings['tokens'] = [tokenizer.id_to_token(i) for i in range(config['vocab_size'])]
    
    print('Training completed.')
    print('Total training time (min): ', (time.time() - time_start)/60)
    print('Total training time (h): ', (time.time() - time_start)/3600)

 
    for key in summary:
        if key != 'embedd_top_tokens':
            summary[key] = np.array(summary[key])
            print(f'{key} : {summary[key].shape}')

    
    save_data(summary,'summary',experiment_name='full_measurements', params=params)
    save_data(data_embeddings,'summary_embeddings',experiment_name='full_measurements', params=params)



if __name__ == "__main__":
    main()