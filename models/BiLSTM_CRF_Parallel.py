from interfaces.model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from TorchCRF import CRF
from typing import List

# Assuming the necessary imports and global constants (like START_TAG, STOP_TAG) are available.

class SelfAttention(nn.Module):
    """Self-attention layer for capturing long-range dependencies."""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        return self.layer_norm(x + self.dropout(attn_out))


class Arabic_BiLSTM_CRF(Model, nn.Module):
    def __init__(self,
                 char_vocab_size: int,
                 num_tags: int,
                 pos_vocab_size: int,
                 char_embedding_dim: int = 256,
                 lstm_hidden_dim: int = 512,
                 fasttext_embedding_dim: int = 300,  # FastText dimension
                 pos_embedding_dim: int = 128,  # POS dimension
                 num_layers: int = 3,
                 dropout: float = 0.4,
                 use_attention: bool = True):  # NEW: Attention flag

        nn.Module.__init__(self)
        Model.__init__(self)

        self.use_attention = use_attention
        
        # Calculate the total input dimension for the LSTM after feature concatenation
        self.lstm_input_dim = char_embedding_dim + fasttext_embedding_dim + pos_embedding_dim

        # 1. Character Embedding Layer (Trainable)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0)
        
        # 2. POS Embedding Layer (Trainable)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embedding_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(dropout)
        
        # 3. Input projection layer (optional, for better feature mixing)
        self.input_proj = nn.Linear(self.lstm_input_dim, self.lstm_input_dim)

        # 4. BiLSTM Layer (The Encoder)
        lstm_dropout = dropout if num_layers > 1 else 0
        self.lstm = nn.LSTM(self.lstm_input_dim,
                            lstm_hidden_dim // 2,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=lstm_dropout)

        # 5. Layer Normalization (Stability)
        self.layer_norm = nn.LayerNorm(lstm_hidden_dim)
        
        # 6. Self-Attention Layer (NEW - for long-range dependencies)
        if use_attention:
            self.self_attention = SelfAttention(lstm_hidden_dim, num_heads=8, dropout=dropout)
        
        # 7. Output projection with residual
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        )
        self.output_norm = nn.LayerNorm(lstm_hidden_dim)

        # 8. Emission Score Projection Layer
        self.hidden2tag = nn.Linear(lstm_hidden_dim, num_tags)

        # 9. CRF Layer (Batched and Efficient)
        self.crf = CRF(num_tags)

    def _get_lstm_features(self, input_ids: torch.Tensor, lengths: torch.Tensor, fasttext_vectors: torch.Tensor, pos_ids: torch.Tensor)-> torch.Tensor:
        """
        Generates emission scores (BiLSTM + Attention output) for the batch.
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. Character Embedding (C_emb)
        char_embedded = self.char_embedding(input_ids)  # (B, L, C_emb_dim)
        
        # 2. POS Embedding
        pos_embedded = self.pos_embedding(pos_ids)  # (B, L, Pos_emb_dim)

        # Apply dropout
        char_embedded = self.input_dropout(char_embedded)
        pos_embedded = self.input_dropout(pos_embedded)

        # 3. Feature Concatenation (C_emb + W + POS)
        lstm_input = torch.cat([char_embedded, fasttext_vectors, pos_embedded], dim=-1)
        
        # 4. Input projection
        lstm_input = self.input_proj(lstm_input)
        
        max_len = lstm_input.size(1)
        lengths_clamped = torch.clamp(lengths, max=max_len)

        # 5. Packing for efficient LSTM
        packed_input = nn.utils.rnn.pack_padded_sequence(
            lstm_input,
            lengths_clamped.cpu().tolist(),
            batch_first=True,
            enforce_sorted=False
        )

        # 6. BiLSTM Pass
        packed_output, _ = self.lstm(packed_input)

        # 7. Unpacking
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=seq_len)

        # 8. Layer Normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # 9. Self-Attention (if enabled)
        if self.use_attention:
            # Create attention mask (True = ignore, False = attend)
            # Use actual lstm_out seq length for mask
            actual_seq_len = lstm_out.size(1)
            lengths_device = lengths.to(input_ids.device)
            attn_mask = torch.arange(actual_seq_len, device=input_ids.device).expand(batch_size, actual_seq_len) >= lengths_device.unsqueeze(1)
            lstm_out = self.self_attention(lstm_out, mask=attn_mask)
        
        # 10. Output projection with residual
        lstm_out = self.output_norm(lstm_out + self.output_proj(lstm_out))

        # 11. Emission Score Projection
        emissions = self.hidden2tag(lstm_out)
        return emissions

    # --- Loss Calculation (using batched CRF) ---
    def neg_log_likelihood(self, input_ids: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor, fasttext_vectors: torch.Tensor, pos_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log likelihood (training loss) using the CRF layer.
        """
        emissions = self._get_lstm_features(input_ids, lengths, fasttext_vectors, pos_ids) # Computes the batched emission scores
        
        # OPTIMIZED: Create mask more efficiently
        batch_size, seq_len = input_ids.shape
        mask = torch.arange(seq_len, device=input_ids.device).expand(batch_size, seq_len) < lengths.unsqueeze(1)

        # CRF computes NLL loss (forward_score - gold_score)
        # TorchCRF (s14t284) seems to expect (batch, seq_len, num_tags) based on error analysis
        # So we do NOT transpose.
        
        # Reduction='mean' applies mean NLL across the batch
        # TorchCRF returns sum log-likelihood by default (check implementation if unsure)
        # We negate it to get NLL.
        ll = self.crf(emissions, labels=labels, mask=mask)
        
        # If TorchCRF returns a vector (per-sequence LL), sum it up
        if ll.dim() > 0:
            ll = ll.sum()
            
        loss = -ll
        
        # Normalize by batch size to mimic reduction='mean'
        return loss / batch_size

    # --- Inference (using batched CRF) ---
    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor, fasttext_vectors: torch.Tensor, pos_ids: torch.Tensor) -> List[List[int]]:
        """
        Inference: return predicted tag sequences via Viterbi decoding.
        """
        emissions = self._get_lstm_features(input_ids, lengths, fasttext_vectors, pos_ids)
        
        # OPTIMIZED: Create mask more efficiently
        batch_size, seq_len = input_ids.shape
        mask = torch.arange(seq_len, device=input_ids.device).expand(batch_size, seq_len) < lengths.unsqueeze(1)

        # Do NOT transpose for TorchCRF
        return self.crf.viterbi_decode(emissions, mask=mask)

# class BiLSTM_CRF(Model, nn.Module):
#     def __init__(self, vocab_size, num_tags, embedding_dim=128, hidden_dim=256, dropout=0.3):
#         super(BiLSTM_CRF, self).__init__()
#         Model.__init__(self)

#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1,
#                            bidirectional=True, batch_first=True, dropout=dropout)
#         self.hidden2tag = nn.Linear(hidden_dim, num_tags)
#         self.crf = CRF(num_tags, batch_first=True)
#         self.dropout = nn.Dropout(dropout)

#     def neg_log_likelihood(self, sentence, tags, mask=None):
#         """Compute negative log likelihood (training loss)"""
#         embeds = self.embedding(sentence)
#         embeds = self.dropout(embeds)
#         lstm_out, _ = self.lstm(embeds)
#         emissions = self.hidden2tag(lstm_out)

#         # CRF computes NLL loss (forward_score - gold_score)
#         loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
#         return loss

#     def forward(self, sentence, mask=None):
#         """Inference: return predicted tag sequences"""
#         embeds = self.embedding(sentence)
#         embeds = self.dropout(embeds)
#         lstm_out, _ = self.lstm(embeds)
#         emissions = self.hidden2tag(lstm_out)

#         # Decode returns list of lists: predicted tags for each batch item
#         predictions = self.crf.decode(emissions, mask=mask)
#         return predictions

# # Usage example
# model = BiLSTM_CRF(vocab_size=10000, num_tags=9).cuda()

# def collate_fn(batch):
#     """Pad sequences and create masks"""
#     sentences, tags = zip(*batch)
#     max_len = max(len(s) for s in sentences)

#     padded_sents = torch.zeros(len(batch), max_len, dtype=torch.long)
#     padded_tags = torch.zeros(len(batch), max_len, dtype=torch.long)
#     masks = torch.zeros(len(batch), max_len, dtype=torch.bool)

#     for i, (sent, tag) in enumerate(zip(sentences, tags)):
#         padded_sents[i, :len(sent)] = torch.tensor(sent)
#         padded_tags[i, :len(tag)] = torch.tensor(tag)
#         masks[i, :len(sent)] = 1

#     return padded_sents, padded_tags, masks

# # Training
# dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# model.train()

# for epoch in range(10):
#     for sentences, tags, masks in dataloader:
#         sentences, tags, masks = sentences.cuda(), tags.cuda(), masks.cuda()

#         optimizer.zero_grad()
#         loss = model.neg_log_likelihood(sentences, tags, masks)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#         optimizer.step()

#     print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# # Inference
# model.eval()
# with torch.no_grad():
#     sentences, tags, masks = next(iter(dataloader))
#     sentences, masks = sentences.cuda(), masks.cuda()
#     predictions = model(sentences, masks)
#     # predictions: list of lists, e.g., [[0,1,2], [1,1,0], ...]

