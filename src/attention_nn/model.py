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
        return self.embedding(x) #* math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    """
    Positional encoding layer to add positional information to token embeddings.
    Args:
        d_model (int): dimension of the embeddings
        seq_len (int): maximum sequence length
        dropout (float): dropout rate 
        type_enc (str): type of positional encoding ('wave' or 'random' or 'learned')
        amp (float): amplitude scaling factor for positional encoding if not 'learned'
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float, type_enc: str = 'wave',amp:float = 1.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        if type_enc == 'random':
            # Random positional encoding
            pe =  torch.randn(1, seq_len, d_model)
            self.register_buffer('pe', amp*pe)
        elif type_enc == 'wave':
            # Sinusoidal positional encoding
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
            pe =  pe.unsqueeze(0) # (1, seq_len, d_model)
            # Register the positional encoding as a buffer
            self.register_buffer('pe', amp*pe)
        elif type_enc == 'learned':
            # Learned positional encoding
            self.pe = nn.Parameter(torch.randn(1, seq_len, d_model))
        else:
            raise ValueError("type_enc must be 'wave', 'random' or 'learned'")

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]  # (batch, seq_len, d_model)
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
        # self.norm = torch.sqrt(torch.arange(1,seq_len+1)).view(1,-1,1)
        # self.register_buffer('norm', torch.sqrt(torch.arange(1,seq_len+1)).view(1,-1,1))
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
            attention_scores: (batch, seq_len, seq_len) x W x.T 
        """
        # x: (batch, seq_len, d_model)
        attention_scores = x @ self.W.weight @ x.transpose(-2, -1) /self.d_model #/ math.sqrt(self.d_model)
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
        attention_scores = proj_left @ proj_right.transpose(-2, -1) / (math.sqrt(self.rank)* self.d_model)
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
        
        attention_probabilities = torch.softmax(attention_scores,dim=-1)  # (batch, seq_len, seq_len)
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
        d (int): dimension of the embeddings
        vocab_size (int): size of the vocabulary
        L (int): maximum sequence length
        dropout (float): dropout rate
        rank (int): rank for low-rank factorization in attention layer
        fr_emb (bool): whether to freeze the embeddings during training
        fr_att (bool): whether to freeze the attention weights during training
        type_enc (str): type of positional encoding ('wave' or 'random' or 'learned')
        skip_residual (bool): whether to skip residual connections
        beta (float): scaling factor for logits
        amp (float): amplitude scaling factor for positional encoding if not 'learned'
        unmb (bool): weather to use different trainable unembedding for last layer or to use transposed embedding matrix
    """
    def __init__(self, 
                d:int, 
                vocab_size:int, 
                L:int, 
                dropout:float=0.1,
                rank:int=None,
                fr_emb:bool=False,
                fr_att:bool=False,
                type_enc:str='wave',
                skip_residual:bool=False,
                beta:float=1.0,
                amp:float=1.0,
                unmb:bool=False
                 ) -> None:
        
        super().__init__()
        self.d=d
        self.vocab_size=vocab_size
        self.L=L
        self.dropout=dropout
        self.rank=rank
        self.fr_emb = fr_emb
        self.fr_att = fr_att
        self.type_enc = type_enc
        self.skip_residual = skip_residual
        self.beta = beta
        self.amp = amp

        # Components
        self.input_embeddings = InputEmbeddings(d,vocab_size,fr_emb)
        self.positional_encoding = PositionalEncoding(d,L,dropout,type_enc,amp)
        self.attention_layer = AttentionLayer(d,L,dropout,rank,fr_att)
        self.residual_connection = ResidualConnection(d,dropout,skip_residual)
        
        
        # If unmb is True, create U matrix for unembedding as parameter
        # else define U as the transpose of the embedding matrix
        if unmb:
            self.U_matrix = nn.Parameter(torch.randn(d, vocab_size))
        else:
            self.U_matrix = None
    def forward(self,input_tokens,attention_mask):
        """ Forward pass of the Transformer model.

        Args:
            input_tokens: (batch, seq_len)
            attention_mask: (batch, seq_len, seq_len)

        Returns:
            logits: (batch, seq_len, vocab_size)"""
        # input_tokens: (batch, seq_len)        
        e = self.input_embeddings(input_tokens)  # (batch, seq_len, d_model)
        x = self.positional_encoding(e)  # (batch, seq_len, d_model)
        y = self.attention_layer(x,attention_mask) # (batch, seq_len, d_model)
        z = self.residual_connection(x, y)  # (batch, seq_len, d_model)
        if self.U_matrix is not None:  # unmb=True
            U = self.U_matrix
        else:  # unmb=False
            U = self.input_embeddings.embedding.weight.t()
        logits = torch.matmul(z, U)  # (batch, seq_len, vocab_size)
        
        return self.beta * logits / math.sqrt(self.d)

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
            - fr_emb (bool): whether to freeze the embeddings during training
            - fr_att (bool): whether to freeze the attention weights during training
            - random_positional_encoding (bool): whether to use random positional encoding
            - skip_residual (bool): whether to skip residual connections
            - type_enc (str): type of positional encoding ('wave' or 'random' or 'learned')
            - beta (float): scaling factor for logits
            - sigma (float): standard deviation for parameter initialization



    Returns:
        model (Transformer): Transformer model instance
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = Transformer(
        d=config.get('d', 512),
        vocab_size=config.get('vocab_size', 1000),
        L=config.get('L', 124),
        dropout=config.get('dropout', 0.0),
        rank=config.get('rank', None),
        fr_emb=config.get('fr_emb', False),
        fr_att=config.get('fr_att', False),
        type_enc=config.get('type_enc', 'wave'),
        skip_residual=config.get('skip_residual', False),
        beta=config.get('beta', 1.0),
        amp=config.get('amp', 1.0),
        unmb=config.get('unmb', False)
        ).to(device)

    # Initialize the parameters
    std_init = config.get('sigma',1.0)#/math.sqrt(config.get('d',512))
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.normal_(param, mean=0.0, std=std_init)
            # torch.nn.init.xavier_normal_(param)    
    return model , device