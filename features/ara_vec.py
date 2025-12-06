"""
Arabic Character Embedding Module for Diacritization

This module provides character-level embeddings for Arabic text.

Input:
    - Arabic character sequences (from ARABIC_LETTERS in constants.py)
    - Each character is represented by its index in the vocabulary
    
Output:
    - Embedding vectors for each character
    - Shape: (batch_size, seq_length, embedding_dim)
    
Usage:
    embedder = ArabicCharEmbedding(vocab_size=100, embedding_dim=128)
    char_ids = torch.tensor([[1, 2, 3, 4, 5]])  # batch of character indices
    embeddings = embedder(char_ids)  # Shape: (1, 5, 128)
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from utils.constants import ARABIC_LETTERS


class ArabicCharEmbedding(nn.Module):
    """
    Character embedding layer for Arabic text.
    
    Converts character indices to dense embedding vectors.
    Includes special tokens: PAD, UNK, BOS, EOS.
    """
    
    # Special token indices
    PAD_IDX = 0
    UNK_IDX = 1
    BOS_IDX = 2
    EOS_IDX = 3
    
    def __init__(
        self,
        embedding_dim: int = 128,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        dropout: float = 0.0
    ):
        """
        Initialize character embedding layer.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            padding_idx: Index for padding token (default: PAD_IDX)
            max_norm: If given, embeddings are normalized to this max norm
            dropout: Dropout probability for embeddings
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else self.PAD_IDX
        self.max_norm = max_norm
        
        # Build character vocabulary
        self.char_to_idx = self._build_vocab()
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=self.padding_idx,
            max_norm=max_norm
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def _build_vocab(self) -> Dict[str, int]:
        """
        Build character vocabulary from ARABIC_LETTERS.
        
        Returns:
            Dictionary mapping characters to indices
        """
        vocab = {
            '<PAD>': self.PAD_IDX,
            '<UNK>': self.UNK_IDX,
            '<BOS>': self.BOS_IDX,
            '<EOS>': self.EOS_IDX,
        }
        
        # Add Arabic letters
        for char in ARABIC_LETTERS:
            if char not in vocab:
                vocab[char] = len(vocab)
        
        return vocab
    
    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: convert character indices to embeddings.
        
        Args:
            char_ids: Character indices tensor
                     Shape: (batch_size, seq_length)
        
        Returns:
            Embedded characters tensor
            Shape: (batch_size, seq_length, embedding_dim)
        """
        # Get embeddings
        embedded = self.embedding(char_ids)
        
        # Apply dropout if configured
        if self.dropout is not None:
            embedded = self.dropout(embedded)
        
        return embedded
    
    def encode_text(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to character indices.
        
        Args:
            text: Input Arabic text
            add_special_tokens: If True, add BOS and EOS tokens
        
        Returns:
            List of character indices
        """
        char_ids = []
        
        if add_special_tokens:
            char_ids.append(self.BOS_IDX)
        
        for char in text:
            char_ids.append(self.char_to_idx.get(char, self.UNK_IDX))
        
        if add_special_tokens:
            char_ids.append(self.EOS_IDX)
        
        return char_ids
    
    def decode_ids(self, char_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert character indices back to text.
        
        Args:
            char_ids: List of character indices
            skip_special_tokens: If True, skip special tokens in output
        
        Returns:
            Decoded text string
        """
        special_tokens = {self.PAD_IDX, self.UNK_IDX, self.BOS_IDX, self.EOS_IDX}
        
        chars = []
        for idx in char_ids:
            if skip_special_tokens and idx in special_tokens:
                continue
            chars.append(self.idx_to_char.get(idx, '<UNK>'))
        
        return ''.join(chars)
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
    
    def get_embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.embedding_dim


class ArabicCharEmbeddingWithPositional(ArabicCharEmbedding):
    """
    Character embedding with positional encoding.
    
    Adds position information to character embeddings using sinusoidal encoding.
    Useful for models that don't have built-in position awareness (e.g., non-recurrent).
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        max_seq_length: int = 512,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        dropout: float = 0.0
    ):
        """
        Initialize character embedding with positional encoding.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            max_seq_length: Maximum sequence length for positional encoding
            padding_idx: Index for padding token
            max_norm: If given, embeddings are normalized to this max norm
            dropout: Dropout probability
        """
        super().__init__(
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            dropout=dropout
        )
        
        self.max_seq_length = max_seq_length
        
        # Create positional encoding
        self.register_buffer(
            'positional_encoding',
            self._create_positional_encoding(max_seq_length, embedding_dim)
        )
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        Create sinusoidal positional encoding.
        
        Args:
            max_len: Maximum sequence length
            d_model: Embedding dimension
        
        Returns:
            Positional encoding tensor (max_len, d_model)
        """
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: convert character indices to embeddings with positional encoding.
        
        Args:
            char_ids: Character indices tensor
                     Shape: (batch_size, seq_length)
        
        Returns:
            Embedded characters with positional encoding
            Shape: (batch_size, seq_length, embedding_dim)
        """
        # Get character embeddings
        embedded = self.embedding(char_ids)
        
        # Add positional encoding
        seq_length = char_ids.size(1)
        if seq_length > self.max_seq_length:
            raise ValueError(
                f"Sequence length {seq_length} exceeds maximum {self.max_seq_length}"
            )
        
        embedded = embedded + self.positional_encoding[:seq_length, :]
        
        # Apply dropout
        if self.dropout is not None:
            embedded = self.dropout(embedded)
        
        return embedded


def create_char_vocabulary() -> Dict[str, int]:
    """
    Create character vocabulary for Arabic diacritization.
    
    Returns:
        Dictionary mapping characters to indices
    """
    embedder = ArabicCharEmbedding()
    return embedder.char_to_idx


def get_vocab_size() -> int:
    """
    Get the vocabulary size for Arabic characters.
    
    Returns:
        Vocabulary size (number of unique characters + special tokens)
    """
    embedder = ArabicCharEmbedding()
    return embedder.vocab_size
