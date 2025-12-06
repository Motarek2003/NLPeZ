from interfaces.model import Model
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from utils.constants import ArabicDiacritics, NUM_DIACRITICS


# CRF special tags
START_TAG = "<START>"
STOP_TAG = "<STOP>"


class DiacritizationDataset(Dataset):
    """
    Dataset for Arabic diacritization.
    
    Each sample contains:
        - char_ids: Character indices (List[int])
        - diacritic_labels: Diacritic labels for each character (List[int])
    """
    
    def __init__(self, char_sequences: List[List[int]], diacritic_labels: List[List[int]]):
        """
        Args:
            char_sequences: List of character ID sequences
            diacritic_labels: List of diacritic label sequences
        """
        assert len(char_sequences) == len(diacritic_labels), \
            "Number of sequences must match number of label sequences"
        
        self.char_sequences = char_sequences
        self.diacritic_labels = diacritic_labels
    
    def __len__(self):
        return len(self.char_sequences)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.char_sequences[idx], dtype=torch.long),
            torch.tensor(self.diacritic_labels[idx], dtype=torch.long)
        )


class BiLSTMCRFDiacritizer(Model, nn.Module):
    """
    BiLSTM-CRF model for Arabic diacritization.
    
    Predicts one of 15 diacritic classes for each Arabic character.
    Uses CRF for sequence-level prediction optimization.
    """
    
    def __init__(
        self,
        char_embedder: nn.Module,
        hidden_dim: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.5,
        num_diacritics: int = NUM_DIACRITICS
    ):
        """
        Initialize BiLSTM-CRF diacritizer.
        
        Args:
            char_embedder: Character embedding module (from ara_vec.py)
            hidden_dim: Hidden dimension for LSTM
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout probability
            num_diacritics: Number of diacritic classes (default: 15)
        """
        nn.Module.__init__(self)
        Model.__init__(self)
        
        self.char_embedder = char_embedder
        self.embedding_dim = char_embedder.get_embedding_dim()
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.num_diacritics = num_diacritics
        
        # Add START and STOP tags for CRF
        self.num_tags = num_diacritics + 2  # +2 for START and STOP
        self.tag_to_idx = self._build_tag_vocab()
        self.idx_to_tag = {idx: tag for tag, idx in self.tag_to_idx.items()}
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            self.embedding_dim,
            hidden_dim // 2,
            num_layers=num_lstm_layers,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=False  # (seq_len, batch, features)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer to map LSTM output to tag scores
        self.hidden2tag = nn.Linear(hidden_dim, self.num_tags)
        
        # CRF transition parameters
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags))
        
        # Initialize transitions: no transition to START, no transition from STOP
        self.transitions.data[self.tag_to_idx[START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_idx[STOP_TAG]] = -10000
    
    def _build_tag_vocab(self):
        """Build tag vocabulary with START and STOP tags."""
        tag_vocab = {}
        
        # Add diacritic tags
        for diacritic in ArabicDiacritics:
            tag_vocab[diacritic.name] = diacritic.value
        
        # Add CRF special tags
        tag_vocab[START_TAG] = self.num_diacritics
        tag_vocab[STOP_TAG] = self.num_diacritics + 1
        
        return tag_vocab
    
    def _get_lstm_features(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Get LSTM features for character sequence.
        
        Args:
            char_ids: Character IDs (seq_length,)
        
        Returns:
            LSTM features (seq_length, num_tags)
        """
        # Get embeddings: (seq_length,) -> (seq_length, 1, embedding_dim)
        embedded = self.char_embedder(char_ids.unsqueeze(1))  # Add batch dimension
        embedded = embedded.squeeze(1)  # Remove batch dimension
        embedded = embedded.unsqueeze(1)  # (seq_length, 1, embedding_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (seq_length, 1, hidden_dim)
        lstm_out = lstm_out.squeeze(1)  # (seq_length, hidden_dim)
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # Linear layer
        lstm_feats = self.hidden2tag(lstm_out)  # (seq_length, num_tags)
        
        return lstm_feats
    
    def _forward_alg(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Forward algorithm to compute partition function (for CRF).
        
        Args:
            feats: LSTM features (seq_length, num_tags)
        
        Returns:
            Log partition function
        """
        # Initialize forward variables
        init_alphas = torch.full((1, self.num_tags), -10000., device=feats.device)
        init_alphas[0][self.tag_to_idx[START_TAG]] = 0.
        forward_var = init_alphas
        
        # Iterate through the sequence
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.num_tags):
                # Broadcast emission score
                emit_score = feat[next_tag].view(1, -1).expand(1, self.num_tags)
                # Transition score
                trans_score = self.transitions[next_tag].view(1, -1)
                # Combine and accumulate
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(torch.logsumexp(next_tag_var, dim=1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        
        # Add transition to STOP
        terminal_var = forward_var + self.transitions[self.tag_to_idx[STOP_TAG]]
        alpha = torch.logsumexp(terminal_var, dim=1)
        
        return alpha
    
    def _score_sentence(self, feats: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        """
        Compute the score of a given tag sequence.
        
        Args:
            feats: LSTM features (seq_length, num_tags)
            tags: Gold tag sequence (seq_length,)
        
        Returns:
            Sequence score
        """
        score = torch.zeros(1, device=feats.device)
        
        # Add START tag
        tags = torch.cat([
            torch.tensor([self.tag_to_idx[START_TAG]], dtype=torch.long, device=tags.device),
            tags
        ])
        
        # Sum transition and emission scores
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        
        # Add transition to STOP
        score = score + self.transitions[self.tag_to_idx[STOP_TAG], tags[-1]]
        
        return score
    
    def _viterbi_decode(self, feats: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Viterbi decoding to find the best tag sequence.
        
        Args:
            feats: LSTM features (seq_length, num_tags)
        
        Returns:
            Tuple of (path_score, best_path)
        """
        backpointers = []
        
        # Initialize viterbi variables
        init_vvars = torch.full((1, self.num_tags), -10000., device=feats.device)
        init_vvars[0][self.tag_to_idx[START_TAG]] = 0
        forward_var = init_vvars
        
        # Forward pass
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            
            for next_tag in range(self.num_tags):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = torch.argmax(next_tag_var).item()
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        
        # Transition to STOP
        terminal_var = forward_var + self.transitions[self.tag_to_idx[STOP_TAG]]
        best_tag_id = torch.argmax(terminal_var).item()
        path_score = terminal_var[0][best_tag_id]
        
        # Backtrack
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        
        # Remove START tag
        start = best_path.pop()
        assert start == self.tag_to_idx[START_TAG], "Path should start with START_TAG"
        best_path.reverse()
        
        return path_score, best_path
    
    def neg_log_likelihood(self, char_ids: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood loss (for training).
        
        Args:
            char_ids: Character IDs (seq_length,)
            tags: Gold diacritic labels (seq_length,)
        
        Returns:
            Negative log-likelihood loss
        """
        feats = self._get_lstm_features(char_ids)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score
    
    def forward(self, char_ids: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Forward pass: predict diacritic sequence.
        
        Args:
            char_ids: Character IDs (seq_length,)
        
        Returns:
            Tuple of (path_score, predicted_tags)
        """
        feats = self._get_lstm_features(char_ids)
        score, tag_seq = self._viterbi_decode(feats)
        return score, tag_seq
    
    def predict(self, char_ids: torch.Tensor) -> List[int]:
        """
        Predict diacritic labels for character sequence.
        
        Args:
            char_ids: Character IDs (seq_length,)
        
        Returns:
            Predicted diacritic labels
        """
        with torch.no_grad():
            _, tag_seq = self.forward(char_ids)
        return tag_seq