from interfaces.model import Model
import torch
import torch.nn as nn
from torchcrf import CRF
from typing import List

# Assuming the necessary imports and global constants (like START_TAG, STOP_TAG) are available.

class Arabic_BiLSTM_CRF(Model, nn.Module):
    def __init__(self,
                 char_vocab_size: int,
                 num_tags: int,
                 char_embedding_dim: int = 128,
                 lstm_hidden_dim: int = 256,
                 fasttext_embedding_dim: int = 100, # FastText dimension
                 dropout: float = 0.3):

        nn.Module.__init__(self)
        Model.__init__(self)

        # Calculate the total input dimension for the LSTM after feature concatenation
        self.lstm_input_dim = char_embedding_dim + fasttext_embedding_dim #char dim + word dim

        # 1. Character Embedding Layer (Trainable)
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0) # Creates a lookup table for all character IDs.
        self.dropout = nn.Dropout(dropout) # Used for regularization, applied to the character embeddings before the LSTM

        # 2. BiLSTM Layer (The Encoder)
        num_layers = 1 # Example
        lstm_dropout = dropout if num_layers > 1 else 0
        self.lstm = nn.LSTM(self.lstm_input_dim,
                            lstm_hidden_dim // 2,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=lstm_dropout) # Dropout applies if num_layers > 1

        # 3. Emission Score Projection Layer
        self.hidden2tag = nn.Linear(lstm_hidden_dim, num_tags)

        # 4. CRF Layer (Batched and Efficient)
        self.crf = CRF(num_tags, batch_first=True)

    def _get_lstm_features(self, input_ids: torch.Tensor, lengths: torch.Tensor, fasttext_vectors: torch.Tensor)-> torch.Tensor:
        """
        Generates emission scores (BiLSTM output) for the batch.
        """
        # 1. Character Embedding (C_emb)
        char_embedded = self.char_embedding(input_ids) # (B, L, C_emb_dim)
        char_embedded = self.dropout(char_embedded)

        # 2. Feature Concatenation (C_emb + W)
        lstm_input = torch.cat([char_embedded, fasttext_vectors], dim=-1) # (B, L, LSTM_input_dim)

        # 3. Packing (Required for performance with variable lengths)
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

        # 6. Emission Score Projection
        emissions = self.hidden2tag(lstm_out) # (B, L, Num_tags)
        # Projects the BiLSTM output to the final tag space. Output shape: $(B, L, \text{Num\_tags})$. These are the final scores fed to the CRF layer.
        return emissions

    # --- Loss Calculation (using batched CRF) ---
    def neg_log_likelihood(self, input_ids: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor, fasttext_vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log likelihood (training loss) using the CRF layer.
        """
        emissions = self._get_lstm_features(input_ids, lengths, fasttext_vectors) # Computes the batched emission scores
        # Create a mask to inform the CRF where the actual sequence ends
        mask = torch.zeros_like(input_ids, dtype=torch.bool).to(input_ids.device) # Initializes a boolean mask tensor.
        for i, length in enumerate(lengths):
            mask[i, :length] = True

        # CRF computes NLL loss (forward_score - gold_score)
        # Reduction='mean' applies mean NLL across the batch
        loss = -self.crf(emissions, tags=labels, mask=mask, reduction='mean')
        return loss

    # --- Inference (using batched CRF) ---
    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor, fasttext_vectors: torch.Tensor) -> List[List[int]]:
        """
        Inference: return predicted tag sequences via Viterbi decoding.
        """
        emissions = self._get_lstm_features(input_ids, lengths, fasttext_vectors)

        # Create mask
        mask = torch.zeros_like(input_ids, dtype=torch.bool).to(input_ids.device)
        for i, length in enumerate(lengths):
            mask[i, :length] = True

        # Decode returns list of lists: predicted tags (IDs) for each batch item
        predictions = self.crf.decode(emissions, mask=mask)
        return predictions

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

