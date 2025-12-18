import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """
    Embedding layer to convert token IDs to dense vectors.
    
    Args:
        d_model (int): dimension of the embeddings
        vocab_size (int): size of the vocabulary 
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    """
    Positional encoding layer to add positional information to token embeddings.
    Args:
        d_model (int): dimension of the embeddings
        seq_len (int): maximum sequence length
        dropout (float): dropout rate 
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
    
class AttentionLayer(nn.Module):
    """
    Self-attention layer with low-rank factorization.
    
    Args:
        d_model (int): dimension of the embeddings
        seq_len (int): maximum sequence length
        dropout (float): dropout rate 
        rank (int): rank for low-rank factorization
    """
    def __init__(self,d_model:int,seq_len:int,dropout:float,rank:int=None) -> None:
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)
        
        # Low-rank factorization: W = Wq @ Wk.T
        # If rank is None, use full rank (original behavior)
        self.rank = rank if rank is not None else d_model
        self.Wq = nn.Linear(d_model, self.rank, bias=False)
        self.Wk = nn.Linear(d_model, self.rank, bias=False)
    
    def forward(self,x,mask):
        # x: (batch, seq_len, d_model)

        # Efficient computation: x @ W @ x.T = (x @ Wq) @ (x @ Wk).T
        # Instead of: x @ W @ x.T
        
        proj_left = self.Wq(x)   # (batch, seq_len, rank) - cheaper!
        proj_right = self.Wk(x)  # (batch, seq_len, rank)
        
        # Compute attention scores efficiently
        attention_scores = proj_left @ proj_right.transpose(-2, -1) / math.sqrt(self.d_model)
        # (batch, seq_len, rank) @ (batch, rank, seq_len) = (batch, seq_len, seq_len)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9) 
        
        attention_scores = torch.softmax(attention_scores, dim=-1)  # (batch, seq_len, seq_len)
        return attention_scores @ x  # (batch, seq_len, d_model)

class SimpleTransformer(nn.Module):
    """
    A simple Transformer model with input embeddings, positional encoding, and self-attention.
    Args:
        d_model (int): dimension of the embeddings
        vocab_size (int): size of the vocabulary
        seq_len (int): maximum sequence length
        dropout (float): dropout rate
        rank (int): rank for low-rank factorization in attention layer
    """
    def __init__(self, d_model:int, vocab_size:int, seq_len:int, dropout:float=0.1,rank:int=None) -> None:
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.seq_len=seq_len
        self.dropout=dropout
        
        self.input_embeddings = InputEmbeddings(d_model,vocab_size)
        self.positional_encoding = PositionalEncoding(d_model,seq_len,dropout)
        self.attention_layer = AttentionLayer(d_model,seq_len,dropout,rank)
    
    def forward(self,encoder_input,attention_mask):
        # encoder_input: (batch, seq_len)        
        x = self.input_embeddings(encoder_input)  # (batch, seq_len, d_model)
        x = self.positional_encoding(x)  # (batch, seq_len, d_model)
        x = self.attention_layer(x,attention_mask) + x # (batch, seq_len, d_model)
        x = torch.matmul(x, self.input_embeddings.embedding.weight.t())  # (batch, seq_len, vocab_size)
        
        return x  # (batch, seq_len, vocab_size)