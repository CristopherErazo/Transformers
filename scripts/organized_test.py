import torch
import torch.nn as nn
import argparse
import time
from tqdm import tqdm
import numpy as np

from attention_nn.model import create_model
from attention_nn.dataset import get_dataloader
from attention_nn.train import  metrics_computations, model_matrices_diagnostics, get_special_batch
from attention_nn.utils import embeddings_computations

from configurations.data_config import make_params_dict, save_data, load_data

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a simple Transformer model.")
    parser.add_argument('--d',type=int,default=256, help='Dimension of the model.')
    parser.add_argument('--dataset_size', type=int, default=1000, help='Size of the dataset.')
    parser.add_argument('--train_fraction', type=float, default=0.9, help='Fraction of data used for training.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--L', type=int, default=124, help='Sequence length.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--rank', type=int, default=64, help='Rank of the model.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate.')
    parser.add_argument('--is_tqdm', type=str,default='True', help='Use tqdm for progress bar?')
    parser.add_argument('--datasource', type=str, default='roneneldan/TinyStories', help='Datasource for the dataset.')
    parser.add_argument('--fr_emb', type=str, default='False', help='Freeze embeddings during training?')
    parser.add_argument('--fr_att', type=str, default='False', help='Freeze attention weights during training?')
    parser.add_argument('--type_enc', type=str, default='learned', help='Type of positional encoding (learned/wave/random).')
    parser.add_argument('--skip_residual', type=str, default='False', help='Skip residual connections?')
    parser.add_argument('--beta', type=float, default=1.0, help='Scaling factor for logits.')
    parser.add_argument('--sigma', type=float, default=1.0, help='Standard deviation for parameter initialization.')
    parser.add_argument('--n_prints', type=int, default=20, help='Number of times to print during training.')
    parser.add_argument('--nprints_matrices', type=int, default=10, help='Number of times to save matrices during training.')

    args = parser.parse_args()
    config = vars(args)
    for key in ['is_tqdm', 'fr_emb', 'fr_att', 'skip_residual']:
        config[key] = True if config[key] == 'True' else False
    
    config['rank'] = None if config['rank'] == -1 else int(config['rank'])
    


    fix_params = { key : config[key] for key in ['rank','d','L','dataset_size','skip_residual']}
    variable_params = { key : config[key] for key in ['batch_size','lr','fr_emb','fr_att','type_enc','beta','sigma']}

    params = {'fixed' : fix_params,
              'variable': variable_params}
    
    # Rescale learning rate
    config['lr'] = config['lr'] / (config['d']**0.5)
    print('Configuration: ', config)

    # Prepare data loaders and tokenizer
    print('Preparing data loaders...')
    train_dataloader, val_dataloader , tokenizer = get_dataloader(config)
    config['vocab_size'] = tokenizer.get_vocab_size()
    print(f'Vocabulary size = {config["vocab_size"]}. Maximum entropy per token = {np.log(config["vocab_size"]):.4f} nats.')
    pad_id = tokenizer.token_to_id("[PAD]")
    special_batch = get_special_batch(train_dataloader,config['L'],k=3)

    # Load data statistics
    names = ['dataset_size','L']
    params_stats = {k: config[k] for k in names}
    data_statistics = load_data('token_counts','data_statistics',params=params_stats,base_dir='./data')
    for key in data_statistics.keys():
        print(f'{key} : {data_statistics[key].shape}')
    
    P_mu = data_statistics['unsorted_token_counts'][1:] # (L-1,V)
    P_tot = P_mu.sum(axis=0) # (V)
    P_mu /= P_mu.sum(axis=-1,keepdims=True)
    P_tot /= P_tot.sum()
    H_mu = -np.nansum(P_mu * np.log(P_mu+ 1e-12), axis=1)  # (L-1,)
    H_tot = -np.nansum(P_tot * np.log(P_tot + 1e-12)) 
    data_stats = (P_mu, P_tot, H_mu, H_tot)

    # Set device and build model
    model , device = create_model(config)
    print(f'Model built on device: {device}')

    # Make each data_stats a tensor and send to device
    data_stats = tuple( torch.tensor(arr, dtype=torch.float32).to(device) for arr in data_stats )
   
    CE_loss = nn.CrossEntropyLoss(ignore_index=pad_id,label_smoothing=0.0)
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],eps=1e-9) 
    # optimizer = torch.optim.SGD(model.parameters(),lr=config['lr'])

    sequence_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]



    summary = {
        'evaluation_steps': [],
        'train_loss': [],
        'val_loss': [],
        'entropy': [],
        'part_ratio': [],
        'pos_attended': [],
        'pred_entropy': [],
        'att_in_fractions': [],
        'KL_uniform': [],
        'KL_tot': [],
        'KL_mu': [],
        'embedd_eigvals': [],
        'embedd_mean': [],
        'embedd_std': [],
        'W_eigvals': [],
        'W_mean': [],
        'W_std': [],
        'sequence_fractions': sequence_fractions,
        'vocab_size': config['vocab_size'],
    }
 
    data_matrices = {
        'matrix_steps': [],
        'A': [],
        'x': [],
        'e': [],
        'tokens_attended': [],
        'embeddings': [],
        'embedd_grad_norms': [],
        'W': [],
        'special_batch': special_batch
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

    nprints_matrices = config['nprints_matrices']
    print_matrices = max(1,tot_global_steps // nprints_matrices)
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
            
            if config['is_tqdm'] : batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Measure validation loss and sparsity metrics
            measure_condition = (global_step % print_every == 0)
            if measure_condition:
                # train_loss , train_entropy , train_pr = metrics_computations(model, train_dataloader, device, CE_loss,sequence_fractions = sequence_fractions)
                val_loss , val_entropy , val_pr , pos_attended, pred_entropy, a_in_fractions, KL_uni, KL_tot, KL_mu = metrics_computations(model, val_dataloader, device, CE_loss,data_stats,sequence_fractions = sequence_fractions)
                summary['evaluation_steps'].append(global_step)
                summary['train_loss'].append(loss.item())
                summary['val_loss'].append(val_loss)
                summary['entropy'].append(val_entropy)
                summary['part_ratio'].append(val_pr)
                summary['pos_attended'].append(pos_attended)
                summary['pred_entropy'].append(pred_entropy)
                summary['att_in_fractions'].append(a_in_fractions)
                summary['KL_uniform'].append(KL_uni)
                summary['KL_tot'].append(KL_tot)
                summary['KL_mu'].append(KL_mu)


                # Record gradient norms
                for name, param in model.named_parameters():
                    short_name = name.split('.')[-2]
                    short_name = f'grad_{short_name}'
                    if param.grad is not None:
                        summary[short_name].append(param.grad.data.norm(2).item())
                    else:
                        summary[short_name].append(0.0)

                # Computation with embeddings
                embeddings = model.input_embeddings.embedding.weight.data.clone() # (vocab_size, d_model)
                eigvals , top_tokens = embeddings_computations(embeddings,tokenizer,m=15,n_tokes=30)
                summary['embedd_eigvals'].append(eigvals)
                summary['embedd_mean'].append( torch.mean(embeddings).item() )
                summary['embedd_std'].append( torch.std(embeddings).item() )

                # summary['embedd_top_tokens'].append(top_tokens)
                
                # Compute and save eigenvalues of W@W^T where W = Wq @ Wk^T 
                # do it in efficient way
                if config['rank'] is None:
                    W = model.attention_layer.W.weight.data.clone() # (d_model, d_model)
                else:
                    Wk = model.attention_layer.Wk.weight.data.clone() # (d_model, rank)
                    Wq = model.attention_layer.Wq.weight.data.clone() # (d_model, rank)
                    W = torch.matmul(Wq, Wk.t()) # (d_model, d_model)

                WWT = torch.matmul(W, W.t())/config['d'] # (d_model, d_model)
                eigvals_WWT = torch.linalg.eigvalsh(WWT).cpu().numpy()
                summary['W_eigvals'].append( eigvals_WWT)
                summary['W_mean'].append( torch.mean(W).item() )
                summary['W_std'].append( torch.std(W).item() )



            # Measure embeddings
            measure_matrices = (global_step % print_matrices == 0) 
            # measure_matrices = global_step in [0,10,20,120,400,500,700]  # For quick testing
            if measure_matrices:
                data_matrices['matrix_steps'].append(global_step)
                # Get gradient of embeddings
                emb_grad = model.input_embeddings.embedding.weight.grad.data.clone() # (vocab_size, d_model)
                # compute its norm per token
                emb_grad_norms = torch.norm(emb_grad, dim=1) # (vocab_size)
                data_matrices['embedd_grad_norms'].append(emb_grad_norms.cpu().numpy())
                
                # Get model matrices diagnostics
                a_save , x_save , e_save , top_k_attended = model_matrices_diagnostics(model,special_batch,device,sequence_fractions=sequence_fractions,k=15)

                data_matrices['A'].append(a_save)
                data_matrices['x'].append(x_save)
                data_matrices['e'].append(e_save)

                # tok_k_attended : (batch_size, len(sequence_fractions), k)
                tokens_attended =[ [ [tokenizer.id_to_token(element) for element in seqfrac_element] 
                                    for seqfrac_element in batch_element ]
                                    for batch_element in top_k_attended ]
                
                data_matrices['tokens_attended'].append(tokens_attended)
                

                embeddings = model.input_embeddings.embedding.weight.data.clone()
                data_matrices['embeddings'].append(embeddings.cpu().numpy())

                W = model.attention_layer.W.weight.data.clone() # (d_model, d_model)
                data_matrices['W'].append(W.cpu().numpy())

                
            global_step += 1
            optimizer.step()
            optimizer.zero_grad()
    
    data_matrices['tokens'] = [tokenizer.id_to_token(i) for i in range(config['vocab_size'])]
    
    print('Training completed.')
    print('Total training time (min): ', (time.time() - time_start)/60)
    print('Total training time (h): ', (time.time() - time_start)/3600)

 
    for key in summary:
        summary[key] = np.array(summary[key])
        print(f'{key} : {summary[key].shape}')

    for key in data_matrices:
        if key != 'tokens' and key != 'tokens_attended' and key != 'special_batch':
            data_matrices[key] = np.array(data_matrices[key])
            print(f'{key} : {data_matrices[key].shape}')
    
    save_data(summary,'summary_unemb',experiment_name='evolution_scalar', params=params)
    save_data(data_matrices,'summary_unemb',experiment_name='evolution_matrices', params=params)
  


if __name__ == "__main__":
    main()