import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def real_mask(ids, lengths):
    """Маска [B, T]: True там, где настоящее слово, False — где дописан <pad>."""
    positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    return positions < lengths.to(ids.device).unsqueeze(1)


def fix_entity_mask(mask, real):
    """Если в примере маска сущности пустая, берём вместо неё все настоящие слова."""
    empty = mask.sum(dim=1) == 0
    return torch.where(empty.unsqueeze(1), real, mask)


def masked_max(h, mask):
    """Максимум по словам, где mask = True. h: [B, T, D], mask: [B, T] -> [B, D]."""
    h = h.masked_fill(~mask.unsqueeze(-1), -1e9)
    return h.max(dim=1).values


def run_lstm(lstm, x, lengths):
    """Прогоняет LSTM только по настоящим словам (packing) и возвращает [B, T, 2H]."""
    packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    out, _ = lstm(packed)
    h, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
    return h


def pool_and_classify(h, real, e1, e2, dropout, out_layer):
    """Склеивает max-pooling по предложению, по e1 и по e2 и выдаёт оценки классов."""
    e1 = fix_entity_mask(e1, real)
    e2 = fix_entity_mask(e2, real)
    sentence_vec = masked_max(h, real)
    e1_vec = masked_max(h, e1)
    e2_vec = masked_max(h, e2)
    features = torch.cat([sentence_vec, e1_vec, e2_vec], dim=1)
    return out_layer(dropout(features))


class BiLSTMClassifier(nn.Module):
    """BiLSTM-классификатор отношений.

    vocab_size - размер словаря, n_classes - число классов (19)."""

    def __init__(self, vocab_size, n_classes, emb_dim=100, hidden=100,
                 dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(3 * 2 * hidden, n_classes)

    def forward(self, ids, lengths, e1, e2):
        real = real_mask(ids, lengths)
        x = self.dropout(self.emb(ids))
        h = run_lstm(self.lstm, x, lengths)
        return pool_and_classify(h, real, e1, e2, self.dropout, self.out)
