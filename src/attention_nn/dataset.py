from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset, DataLoader , random_split
import torch
import numpy as np


def get_all_sentences(raw_dataset):
    """
    Generator to yield all sentences from the dataset.
    Args:
        raw_dataset : Dataset object with 'text' field
    Yields:
        str : text of each item in the dataset
    """
    for item in raw_dataset:
        yield item['text']

def build_tokenizer(raw_dataset,min_frec:int=2) -> Tokenizer:
    """ Builds a WordLevel tokenizer from the raw dataset.
    Args:
        raw_dataset: Dataset object with 'text' field
    Returns:
        Tokenizer: trained WordLevel tokenizer
    """
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=min_frec)
    tokenizer.train_from_iterator(get_all_sentences(raw_dataset), trainer=trainer)
    return tokenizer

class NextTokenDataset(Dataset):
    """
    Dataset for next-token prediction from a single text source.
    
    Each item is a dictionary with:
        input: (L) tensor of token IDs as model input
        label: (L) tensor of token IDs as target output
        attention_mask: (L, L) boolean tensor for attention masking
        text: original text string for debugging

    Args:
        raw_dataset: Dataset, object with 'text' field
        tokenizer: Tokenizer, object for tokenizing text
        L (int): fixed sequence length for input/output
    """
    
    def __init__(self, raw_dataset, tokenizer:Tokenizer, L:int) -> None:

        super().__init__()
        self.ds = raw_dataset
        self.tokenizer = tokenizer
        self.L = L
        
        # Get special token IDs
        self.sos_id = tokenizer.token_to_id("[SOS]")
        self.eos_id = tokenizer.token_to_id("[EOS]")
        self.pad_id = tokenizer.token_to_id("[PAD]")
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        text = self.ds[idx]['text']
        
        # Tokenize text
        tokens = self.tokenizer.encode(text).ids
        
        # Skip if too short
        if len(tokens) < 10:
            return None
        
        # Truncate if too long (with room for SOS and EOS)
        max_content_len = self.L - 1
        if len(tokens) > max_content_len:
            tokens = tokens[:max_content_len]
        
        # Create input and target sequences
        # Input: [SOS] + tokens + [PAD]*
        # Target: tokens + [EOS] + [PAD]*
        num_padding = self.L - len(tokens) - 1
        
        if num_padding < 0:
            raise ValueError(f"Text too long: {len(tokens)} tokens > {max_content_len}")
        
        # Input sequence (context): [SOS] + text_tokens + [PAD]*
        input = torch.cat([
            torch.tensor([self.sos_id], dtype=torch.int64),
            torch.tensor(tokens, dtype=torch.int64),
            torch.tensor([self.pad_id] * num_padding, dtype=torch.int64),
        ])
        
        # Target sequence (what we want to predict): text_tokens + [EOS] + [PAD]*
        label = torch.cat([
            torch.tensor(tokens, dtype=torch.int64),
            torch.tensor([self.eos_id], dtype=torch.int64),
            torch.tensor([self.pad_id] * (num_padding), dtype=torch.int64),
        ])
        
        # pad mask: True for real tokens, False for PAD
        pad_mask = (input != self.pad_id).unsqueeze(-1)  # ( L, 1) boolean
        pad_mask = pad_mask.bool()

        # causal mask: lower triangular (allow attend to <= position)
        causal_mask = torch.tril(torch.ones((self.L, self.L), dtype=torch.bool))  # ( L, L)

        # combine to (1, L, L): allow only real tokens and past
        attention_mask = pad_mask & causal_mask  # (1, L, L), boolean
            
        assert input.size(0) == self.L
        assert label.size(0) == self.L
        
        return {
            "input": input,                      # (L): context to feed to model
            "label": label,                      # (L): target next-token sequence
            "attention_mask": attention_mask,    # (1, L)&( L, L) pad and causal mask 
            "text": text,                        # (str): original text for debugging
            "tokens_len" : len(tokens)                    # (int): length of original tokenized text
        }
    
# After splitting raw_train_ds and raw_val_ds
def filter_short(ds, tokenizer, min_len=10):
    return [item for item in ds if len(tokenizer.encode(item['text']).ids) >= min_len]


def get_dataloader(config:dict) -> tuple[DataLoader,DataLoader,Tokenizer]:
    """
    Creates training and validation DataLoaders from the dataset.
    Args:
        config (dict): dictionary with configuration parameters:
            - datasource: str, name of the dataset to load (e.g., "roneneldan/TinyStories")
            - dataset_size: int, number of samples to load from the dataset
            - train_fraction: float, fraction of data to use for training
            - L: int, fixed sequence length for input/output
            - batch_size: int, batch size for DataLoader
    Returns:
        train_dataloader: DataLoader for training set
        val_dataloader: DataLoader for validation set
        tokenizer: Tokenizer object used for tokenization
    """
    # Load the dataset 
    raw_dataset = load_dataset(config['datasource'], split='train[:{}]'.format(config['dataset_size']))
    # Build tokenizer
    tokenizer = build_tokenizer(raw_dataset)
    
    # Split dataset into train and validation sets
    train_size = int(config['train_fraction'] * len(raw_dataset))
    val_size = len(raw_dataset) - train_size
    raw_train_ds , raw_val_ds = random_split(raw_dataset,[train_size,val_size])
    
    # Filter out short sequences
    
    raw_train_ds = filter_short(raw_train_ds, tokenizer)
    raw_val_ds = filter_short(raw_val_ds, tokenizer)
    
    # Create  NextTokenDataset instances
    train_dataset = NextTokenDataset(raw_train_ds,tokenizer,config['L'])
    val_dataset = NextTokenDataset(raw_val_ds,tokenizer,config['L'])

    # Find the maximum length of sentence in the dataset
    max_len = 0
    min_len = 1e6
    for item in raw_dataset:
        ids = tokenizer.encode(item['text']).ids
        max_len = max(max_len,len(ids))
        min_len = min(min_len,len(ids))
    print(f'Min sequence length = {min_len}')
    print(f'Max sequence length = {max_len}')

    train_dataloader = DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True)
    val_dataloader = DataLoader(val_dataset,batch_size=config['batch_size'],shuffle=True)
    return train_dataloader, val_dataloader , tokenizer



### Random Dataset

def random_dataset(vocab_size:int, P:np.array, dataset_size:int,seq_len:int) -> list[dict]:
    """
    Creates a random dataset of text samples based on a given vocabulary size and probability distribution.
    Args:
        vocab_size (int): size of the vocabulary
        P (np.array): probability distribution over the vocabulary
        dataset_size (int): number of samples to generate
        seq_len (int): fixed length of each text sample
    Returns:
        list of dict: each dict has a 'text' field with the generated text sample
    """
    
    dataset = []
    values = np.arange(vocab_size) # to avoid special tokens
    for _ in range(dataset_size):
        tokens = np.random.choice(values, size=seq_len+1, p=P)
        dataset.append(list(tokens))
    return dataset

class RandomNextTokenDataset(Dataset):
    """
    Dataset for next-token prediction from a single text source.
    
    Each item is a dictionary with:
        input: (L) tensor of token IDs as model input
        label: (L) tensor of token IDs as target output
        attention_mask: (L, L) boolean tensor for attention masking
        text: original text string for debugging

    Args:
        raw_dataset: Dataset, object with 'text' field
        tokenizer: Tokenizer, object for tokenizing text
        L (int): fixed sequence length for input/output
    """
    
    def __init__(self, raw_dataset, L:int) -> None:

        super().__init__()
        self.ds = raw_dataset
        self.L = L
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        # Token sequence
        tokens = self.ds[idx]
        
        # Input sequence (context): tokens up to L-1 
        input = torch.tensor(tokens[:-1], dtype=torch.int64)
        
        # Target sequence (what we want to predict): shifted by one 
        label = torch.tensor(tokens[1:], dtype=torch.int64)
        
        # causal mask: lower triangular (allow attend to <= position)
        causal_mask = torch.tril(torch.ones((self.L, self.L), dtype=torch.bool))  # ( L, L)
  
        assert input.size(0) == self.L
        assert label.size(0) == self.L
        
        return {
            "input": input,                      # (L): context to feed to model
            "label": label,                      # (L): target next-token sequence
            "attention_mask": causal_mask,    # ( L, L) pad and causal mask 
            "tokens_len" : len(tokens)                    # (int): length of original tokenized text 
        }
    

def get_random_dataloader(config:dict) -> tuple[DataLoader,DataLoader,Tokenizer]:
    """
    Creates training and validation DataLoaders from the dataset.
    Args:
        config (dict): dictionary with configuration parameters:
            - vocab_size: int, size of the vocabulary
            - P: np.array, probability distribution over the vocabulary
            - dataset_size: int, number of samples to load from the dataset
            - train_fraction: float, fraction of data to use for training
            - L: int, fixed sequence length for input/output
            - batch_size: int, batch size for DataLoader
    Returns:
        train_dataloader: DataLoader for training set
        val_dataloader: DataLoader for validation set
        tokenizer: Tokenizer object used for tokenization
    """
    # Create the dataset
    raw_dataset = random_dataset(config['vocab_size'], config['P_tokens'], config['dataset_size'], config['L'])
    
    # Split dataset into train and validation sets
    train_size = int(config['train_fraction'] * len(raw_dataset))
    val_size = len(raw_dataset) - train_size
    raw_train_ds , raw_val_ds = random_split(raw_dataset,[train_size,val_size])
    
 
    # Create  NextTokenDataset instances
    train_dataset = RandomNextTokenDataset(raw_train_ds,config['L'])
    val_dataset = RandomNextTokenDataset(raw_val_ds,config['L'])


    train_dataloader = DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True)
    val_dataloader = DataLoader(val_dataset,batch_size=config['batch_size'],shuffle=True)
    return train_dataloader, val_dataloader

