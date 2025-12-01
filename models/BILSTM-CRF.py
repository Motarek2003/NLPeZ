from interfaces.model import Model

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Sample data
sentences = [["I", "love", "PyTorch"], ["This", "is", "a", "test"]]
tags = [["O", "O", "B - TECH"], ["O", "O", "O", "O"]]

# Create vocabulary for words and tags
word_to_idx = {}
tag_to_idx = {}
for sent in sentences:
    for word in sent:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
for tag_seq in tags:
    for tag in tag_seq:
        if tag not in tag_to_idx:
            tag_to_idx[tag] = len(tag_to_idx)
# Add CRF special tags
START_TAG = "<START>"
STOP_TAG = "<STOP>"

tag_to_idx[START_TAG] = len(tag_to_idx)
tag_to_idx[STOP_TAG]  = len(tag_to_idx)

# Convert sentences and tags to indices
sentences_idx = [[word_to_idx[word] for word in sent] for sent in sentences]
tags_idx = [[tag_to_idx[tag] for tag in tag_seq] for tag_seq in tags]

class SeqDataset(Dataset):
    def __init__(self, sentences, tags):
        self.sentences = sentences
        self.tags = tags

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx]), torch.tensor(self.tags[idx])


dataset = SeqDataset(sentences_idx, tags_idx)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

class BiLSTM_CRF(Model, nn.Module):
    def __init__(self, vocab_size, tag_to_idx, embedding_dim, hidden_dim):
        nn.Module.__init__(self)     # must be first
        Model.__init__(self)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, len(tag_to_idx))
        self.transitions = nn.Parameter(torch.randn(len(tag_to_idx), len(tag_to_idx)))
        self.tag_to_idx = tag_to_idx

    def _forward_alg(self, feats):
        # Forward algorithm to calculate the partition function
        init_alphas = torch.full((1, len(self.tag_to_idx)), -10000.)
        init_alphas[0][self.tag_to_idx["<START>"]] = 0.
        forward_var = init_alphas
        for feat in feats:
            alphas_t = []
            for next_tag in range(len(self.tag_to_idx)):
                emit_score = feat[next_tag].view(1, -1).expand(1, len(self.tag_to_idx))
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(torch.logsumexp(next_tag_var, dim=1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_idx["<STOP>"]]
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    def _get_lstm_features(self, sentence):
        embedded = self.embedding(sentence).view(len(sentence), 1, -1)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out.view(len(sentence), -1)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Calculate the score of a given tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_idx["<START>"]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_idx["<STOP>"], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        init_vvars = torch.full((1, len(self.tag_to_idx)), -10000.)
        init_vvars[0][self.tag_to_idx["<START>"]] = 0
        forward_var = init_vvars

        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(len(self.tag_to_idx)):
                # next_tag_var: (1, num_tags)
                next_tag_var = forward_var + self.transitions[next_tag]
                # best_tag_id as Python int
                best_tag_id = torch.argmax(next_tag_var).item()
                bptrs_t.append(best_tag_id)  # store int
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_idx["<STOP>"]]
        best_tag_id = torch.argmax(terminal_var).item()
        path_score = terminal_var[0][best_tag_id]

        # Backtrack
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.tag_to_idx["<START>"]
        best_path.reverse()
        return path_score, best_path  # best_path is now List[int]


    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

EMBEDDING_DIM = 5
HIDDEN_DIM = 4
model = BiLSTM_CRF(len(word_to_idx), tag_to_idx, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

for epoch in range(300):
    for sentence, tags in dataloader:
        model.zero_grad()
        loss = model.neg_log_likelihood(sentence[0], tags[0])
        loss.backward()
        optimizer.step()

from sklearn.metrics import classification_report

all_preds = []
all_true = []
with torch.no_grad():
    for sentence, tags in dataloader:
        _, pred_tags = model(sentence[0])
        all_preds.extend(pred_tags)
        all_true.extend(tags[0].tolist())

idx_to_tag = {v: k for k, v in tag_to_idx.items()}
pred_tags_str = [idx_to_tag[idx] for idx in all_preds]
true_tags_str = [idx_to_tag[idx] for idx in all_true]
print(classification_report(true_tags_str, pred_tags_str))