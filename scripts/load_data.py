import torch 
from torch.utils.tensorboard import SummaryWriter
from attention_nn.utils import get_weights_file_path 
from configurations.data_config import read_config
from attention_nn.dataset import get_dataloader 
import argparse
import numpy as np


def main():

    parser = argparse.ArgumentParser(description="Train a simple Transformer model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    # Load configuration
    config_path = args.config
    # Read the configuration file
    config = read_config(config_path)
     # Prepare data loaders and tokenizer
    print('Preparing data loaders...')
    _, _ , tokenizer = get_dataloader(config)
    experiment_name = 'logs/simple_experiment/embeddings'
    writter = SummaryWriter(log_dir=experiment_name,)  
    epochs = [0,10,20,30,90]
    # Fraction of embeddings to write
    fraction = 0.1
    vocab_size = tokenizer.get_vocab_size()
    num_embeddings_to_write = int(vocab_size * fraction)
    for epoch in epochs:
        epoch = f'{epoch:02d}'
        print(f'Epoch {epoch}')
        model_filename = get_weights_file_path(config, epoch)
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model_state_dict = state['model_state_dict']
        embeddings = model_state_dict['input_embeddings.embedding.weight']
        print('Embeddings shape:', embeddings.shape)
        writter.add_embedding(embeddings[:num_embeddings_to_write], metadata=[tokenizer.id_to_token(i) for i in range(num_embeddings_to_write)], tag=f'epoch_{epoch}')
        writter.flush()
    writter.close()


if __name__ == "__main__":
    main()