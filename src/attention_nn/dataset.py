from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset, DataLoader , random_split
import torch

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

def build_tokenizer(raw_dataset):
    """ Builds a WordLevel tokenizer from the raw dataset.
    Args:
        raw_dataset: Dataset object with 'text' field
    Returns:
        Tokenizer: trained WordLevel tokenizer
    """
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
    tokenizer.train_from_iterator(get_all_sentences(raw_dataset), trainer=trainer)
    return tokenizer

class NextTokenDataset(Dataset):
    """
    Dataset for next-token prediction from a single text source.
    
    Each item is a dictionary with:
        input: (seq_len) tensor of token IDs as model input
        label: (seq_len) tensor of token IDs as target output
        attention_mask: (seq_len, seq_len) boolean tensor for attention masking
        text: original text string for debugging

    Args:
        raw_dataset: Dataset, object with 'text' field
        tokenizer: Tokenizer, object for tokenizing text
        seq_len (int): fixed sequence length for input/output
    """
    
    def __init__(self, raw_dataset, tokenizer:Tokenizer, seq_len:int) -> None:

        super().__init__()
        self.ds = raw_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
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
        if len(tokens) < 2:
            return None
        
        # Truncate if too long (with room for SOS and EOS)
        max_content_len = self.seq_len - 1
        if len(tokens) > max_content_len:
            tokens = tokens[:max_content_len]
        
        # Create input and target sequences
        # Input: [SOS] + tokens + [PAD]*
        # Target: tokens + [EOS] + [PAD]*
        num_padding = self.seq_len - len(tokens) - 1
        
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
        pad_mask = (input != self.pad_id).unsqueeze(-1)  # ( seq_len, 1) boolean
        pad_mask = pad_mask.bool()

        # causal mask: lower triangular (allow attend to <= position)
        causal_mask = torch.tril(torch.ones((self.seq_len, self.seq_len), dtype=torch.bool))  # ( seq_len, seq_len)

        # combine to (1, seq_len, seq_len): allow only real tokens and past
        attention_mask = pad_mask & causal_mask  # (1, seq_len, seq_len), boolean
            
        assert input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return {
            "input": input,                      # (seq_len): context to feed to model
            "label": label,                      # (seq_len): target next-token sequence
            "attention_mask": attention_mask,    # (1, seq_len)&( seq_len, seq_len) pad and causal mask 
            "text": text,                        # (str): original text for debugging
        }
    

def get_dataloader(config:dict) -> tuple[DataLoader,DataLoader,Tokenizer]:
    """
    Creates training and validation DataLoaders from the dataset.
    Args:
        config (dict): dictionary with configuration parameters:
            - datasource: str, name of the dataset to load (e.g., "roneneldan/TinyStories")
            - dataset_size: int, number of samples to load from the dataset
            - train_fraction: float, fraction of data to use for training
            - seq_len: int, fixed sequence length for input/output
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
    # Create  NextTokenDataset instances
    train_dataset = NextTokenDataset(raw_train_ds,tokenizer,config['seq_len'])
    val_dataset = NextTokenDataset(raw_val_ds,tokenizer,config['seq_len'])

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

