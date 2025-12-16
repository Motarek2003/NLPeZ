from interfaces.model import Model
import torch
import torch.nn as nn
from TorchCRF import CRF
from typing import List

# Assuming the necessary imports and global constants (like START_TAG, STOP_TAG) are available.

class Arabic_BiLSTM_CRF(Model, nn.Module):
    def __init__(self,
                 char_vocab_size: int,
                 num_tags: int,
                 pos_vocab_size: int,
                 char_embedding_dim: int = 128,
                 lstm_hidden_dim: int = 256,
                 fasttext_embedding_dim: int = 100, # FastText dimension
                 pos_embedding_dim: int = 32, # POS dimension
                 num_layers: int = 2,
                 dropout: float = 0.5):

        nn.Module.__init__(self)
        Model.__init__(self)

        # Calculate the total input dimension for the LSTM after feature concatenation
        self.lstm_input_dim = char_embedding_dim + fasttext_embedding_dim + pos_embedding_dim #char dim + word dim + pos dim

        # 1. Character Embedding Layer (Trainable)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0) # Creates a lookup table for all character IDs.
        
        # 2. POS Embedding Layer (Trainable)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embedding_dim, padding_idx=0)

        self.dropout = nn.Dropout(dropout) # Used for regularization, applied to the character embeddings before the LSTM

        # 3. BiLSTM Layer (The Encoder)
        lstm_dropout = dropout if num_layers > 1 else 0
        self.lstm = nn.LSTM(self.lstm_input_dim,
                            lstm_hidden_dim // 2,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=lstm_dropout) # Dropout applies if num_layers > 1

        # 3.5 Layer Normalization (Stability)
        self.layer_norm = nn.LayerNorm(lstm_hidden_dim)

        # 3. Emission Score Projection Layer
        self.hidden2tag = nn.Linear(lstm_hidden_dim, num_tags)

        # 4. CRF Layer (Batched and Efficient)
        # Adapted for TorchCRF (s14t284) which does not support batch_first=True in __init__
        self.crf = CRF(num_tags)

    def _get_lstm_features(self, input_ids: torch.Tensor, lengths: torch.Tensor, fasttext_vectors: torch.Tensor, pos_ids: torch.Tensor)-> torch.Tensor:
        """
        Generates emission scores (BiLSTM output) for the batch.
        """
        # 1. Character Embedding (C_emb)
        char_embedded = self.char_embedding(input_ids) # (B, L, C_emb_dim)
        
        # 2. POS Embedding
        pos_embedded = self.pos_embedding(pos_ids) # (B, L, Pos_emb_dim)

        # Apply dropout
        char_embedded = self.dropout(char_embedded)
        pos_embedded = self.dropout(pos_embedded)

        # 3. Feature Concatenation (C_emb + W + POS)
        lstm_input = torch.cat([char_embedded, fasttext_vectors, pos_embedded], dim=-1) # (B, L, LSTM_input_dim)
        max_len = lstm_input.size(1)
        lengths = torch.clamp(lengths, max=max_len)

        # 4. Packing (Required for performance with variable lengths)
        # .cpu().tolist() is necessary here
        packed_input = nn.utils.rnn.pack_padded_sequence(
            lstm_input,
            lengths.cpu().tolist(),
            batch_first=True,
            enforce_sorted=False
        )# It converts the padded batch into a single, contiguous tensor, ignoring the padding tokens.

        # 4. BiLSTM Pass
        packed_output, _ = self.lstm(packed_input) # Runs the packed sequence through the LSTM.

        # 5. Unpacking
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # Converts the LSTM output back into a padded tensor format, ensuring the output aligns with the original batch shape $(B, L, H_{dim})$

        # Apply LayerNorm
        lstm_out = self.layer_norm(lstm_out)

        # 6. Emission Score Projection
        emissions = self.hidden2tag(lstm_out) # (B, L, Num_tags)
        # Projects the BiLSTM output to the final tag space. Output shape: $(B, L, \text{Num\_tags})$. These are the final scores fed to the CRF layer.
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

