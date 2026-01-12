import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """
    Embedding layer to convert token IDs to dense vectors.
    
    Args:
        d_model (int): dimension of the embeddings
        vocab_size (int): size of the vocabulary 
        freeze (bool): whether to freeze the embeddings during training
    """
    def __init__(self, d_model: int, vocab_size: int, freeze:bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        if freeze:
            self.embedding.weight.requires_grad = False

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
        amplitude (float): amplitude of the positional encoding
        random (bool): whether to use random positional encoding
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float, amplitude: float = 1.0, random: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        if random:
            # Random positional encoding
            pe = amplitude * torch.randn(1, seq_len, d_model)
            self.register_buffer('pe', pe)
        else:
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
            pe = amplitude * pe.unsqueeze(0) # (1, seq_len, d_model)
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
        freeze (bool): whether to freeze the attention weights during training
    """
    def __init__(self,d_model:int,seq_len:int,dropout:float,rank:int=None,freeze:bool=False) -> None:
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)
        
        self.rank = rank
        # If rank is None, use full rank (original behavior) with single matrix W
        if rank is None:
            self.W = nn.Linear(d_model, d_model, bias=False)
            if freeze:
                self.W.weight.requires_grad = False
            self.dot = self.dot_full
        else:
              # Low-rank factorization: W = Wq @ Wk.T
            self.Wq = nn.Linear(d_model, self.rank, bias=False)
            self.Wk = nn.Linear(d_model, self.rank, bias=False)
            if freeze:
                self.Wq.weight.requires_grad = False
                self.Wk.weight.requires_grad = False
            self.dot = self.dot_lowrank


    def dot_full(self,x):
        """ Compute attention scores using full weight matrix.
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            attention_scores: (batch, seq_len, seq_len) x.T W x
        """
        # x: (batch, seq_len, d_model)
        attention_scores = x @ self.W.weight @ x.transpose(-2, -1) / math.sqrt(self.d_model)
        # (batch, seq_len, d_model) @ (d_model, d_model) @ (batch, d_model, seq_len) = (batch, seq_len, seq_len)
        return attention_scores 
    
    def dot_lowrank(self,x):
        """ Compute attention scores using low-rank factorization.
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            attention_scores: (batch, seq_len, seq_len) x.T W x
        """
        # x: (batch, seq_len, d_model)

        # Efficient computation: x @ W @ x.T = (x @ Wq) @ (x @ Wk).T
        # Instead of: x @ W @ x.T
        
        proj_left = self.Wq(x)   # (batch, seq_len, rank) - cheaper!
        proj_right = self.Wk(x)  # (batch, seq_len, rank)
        
        # Compute attention scores efficiently
        attention_scores = proj_left @ proj_right.transpose(-2, -1) / math.sqrt(self.rank)
        # (batch, seq_len, rank) @ (batch, rank, seq_len) = (batch, seq_len, seq_len)
        return attention_scores
    
    
    def attention_probabilities(self, x, mask):
        """ Compute attention scores using low-rank factorization.
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len)
        Returns:
            attention_probabilities: (batch, seq_len, seq_len)
        """
        # x: (batch, seq_len, d_model)

        
        # Compute attention scores efficiently
        attention_scores = self.dot(x)  # (batch, seq_len, seq_len)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9) 
        
        attention_probabilities = torch.softmax(attention_scores, dim=-1)  # (batch, seq_len, seq_len)
        return attention_probabilities

    def forward(self,x,mask):
        # x: (batch, seq_len, d_model)
        attention_probabilities = self.attention_probabilities(x, mask)  # (batch, seq_len, seq_len)
        attention_probabilities = self.dropout(attention_probabilities)
        return attention_probabilities @ x  # (batch, seq_len, d_model)


class ResidualConnection(nn.Module):
    """
    Residual connection layer.
    Args:
        d_model (int): dimension of the embeddings
        dropout (float): dropout rate 
        skip_residual (bool): whether to skip residual connections
    """
    def __init__(self, d_model: int, dropout: float, skip_residual: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        if skip_residual: # If skip_residual is True, do not apply residual connection
            self.forward = self.skip_forward    
    def skip_forward(self, x, sublayer_output):
        return sublayer_output  # (batch, seq_len, d_model)

    def forward(self, x, sublayer_output):
        # x: (batch, seq_len, d_model)
        # sublayer_output: (batch, seq_len, d_model)
        return x + sublayer_output  # (batch, seq_len, d_model)


class Transformer(nn.Module):
    """
    A simple Transformer model with input embeddings, positional encoding, and self-attention.
    Args:
        d_model (int): dimension of the embeddings
        vocab_size (int): size of the vocabulary
        seq_len (int): maximum sequence length
        dropout (float): dropout rate
        rank (int): rank for low-rank factorization in attention layer
        freeze_embeddings (bool): whether to freeze the embeddings during training
        freeze_attention (bool): whether to freeze the attention weights during training
        random_positional_encoding (bool): whether to use random positional encoding
        skip_residual (bool): whether to skip residual connections
        amplitude (float): amplitude of the positional encoding
    """
    def __init__(self, 
                d_model:int, 
                vocab_size:int, 
                seq_len:int, 
                dropout:float=0.1,
                rank:int=None,
                freeze_embeddings:bool=False,
                freeze_attention:bool=False,
                random_positional_encoding:bool=False,
                skip_residual:bool=False,
                amplitude:float=1.0
                 ) -> None:
        
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.seq_len=seq_len
        self.dropout=dropout
        self.rank=rank
        self.freeze_embeddings=freeze_embeddings
        self.freeze_attention=freeze_attention
        self.random_positional_encoding=random_positional_encoding
        self.skip_residual=skip_residual
        self.amplitude=amplitude
        

        # Components
        self.input_embeddings = InputEmbeddings(d_model,vocab_size,freeze_embeddings)
        self.positional_encoding = PositionalEncoding(d_model,seq_len,dropout,amplitude,random_positional_encoding)
        self.attention_layer = AttentionLayer(d_model,seq_len,dropout,rank,freeze_attention)
        self.residual_connection = ResidualConnection(d_model,dropout,skip_residual)

        
    def forward(self,input_tokens,attention_mask):
        """ Forward pass of the Transformer model.

        Args:
            input_tokens: (batch, seq_len)
            attention_mask: (batch, seq_len, seq_len)

        Returns:
            logits: (batch, seq_len, vocab_size)"""
        # input_tokens: (batch, seq_len)        
        x = self.input_embeddings(input_tokens)  # (batch, seq_len, d_model)
        x = self.positional_encoding(x)  # (batch, seq_len, d_model)
        a = self.attention_layer(x,attention_mask) # (batch, seq_len, d_model)
        x = self.residual_connection(x, a)  # (batch, seq_len, d_model)
        x = torch.matmul(x, self.input_embeddings.embedding.weight.t()) # (batch, seq_len, vocab_size)

        return x  # Logits: (batch, seq_len, vocab_size)
    

def create_model(config: dict) -> Transformer:
    """
    Create a Transformer model from a configuration dictionary.
    
    Args:
        config (dict): configuration dictionary with model parameters   
            - d_model (int): dimension of the embeddings
            - vocab_size (int): size of the vocabulary
            - seq_len (int): maximum sequence length
            - dropout (float): dropout rate
            - rank (int): rank for low-rank factorization in attention layer
            - freeze_embeddings (bool): whether to freeze the embeddings during training
            - freeze_attention (bool): whether to freeze the attention weights during training
            - random_positional_encoding (bool): whether to use random positional encoding
            - skip_residual (bool): whether to skip residual connections
            - amplitude (float): amplitude of the positional encoding
            - sigma (float): standard deviation for parameter initialization


    Returns:
        model (Transformer): Transformer model instance
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = Transformer(
        d_model=config.get('d_model', 128),
        vocab_size=config.get('vocab_size', 1000),
        seq_len=config.get('seq_len', 50),
        dropout=config.get('dropout', 0.1),
        rank=config.get('rank', None),
        freeze_embeddings=config.get('freeze_embeddings', False),
        freeze_attention=config.get('freeze_attention', False),
        random_positional_encoding=config.get('random_positional_encoding', False),
        skip_residual=config.get('skip_residual', False),
        amplitude=config.get('amplitude', 1.0)
    ).to(device)

    # Initialize the parameters
    std_init = config.get('sigma',1.0) / (model.d_model ** 0.5)
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.normal_(param, mean=0.0, std=std_init)    
    return model , device