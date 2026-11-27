import torch
import torch.nn as nn

from src.relation_extraction.nn.bilstm import (
    make_embedding,
    real_mask,
    run_lstm,
    pool_and_classify,
)


class GCNClassifier(nn.Module):
    """GCN-классификатор отношений по дереву зависимостей (идея Zhang et al., 2018).

    Эмбеддинги -> BiLSTM для контекста (можно выключить) -> n_layers слоёв GCN ->
    max-pooling по предложению, e1 и e2, как у BiLSTMClassifier -> 19 классов.
    Слой GCN: h = ReLU(W * ((A + I) h) / степень), где A - матрица смежности
    из collate_gcn, I - петли на настоящих словах.

    use_lstm=True - вариант C-GCN, use_lstm=False - GCN прямо на эмбеддингах."""

    def __init__(self, vocab_size, n_classes, emb_dim=100, hidden=100,
                 n_layers=2, dropout=0.5, embeddings=None, use_lstm=True):
        super().__init__()
        self.emb = make_embedding(vocab_size, emb_dim, embeddings)
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
            in_dim = 2 * hidden
        else:
            self.lstm = None
            in_dim = emb_dim

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.layers.append(nn.Linear(in_dim, hidden))
            else:
                self.layers.append(nn.Linear(hidden, hidden))

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(3 * hidden, n_classes)

    def forward(self, ids, lengths, e1, e2, adj=None):
        if adj is None:
            raise ValueError("GCNClassifier needs adjacency matrices: use collate_gcn")

        real = real_mask(ids, lengths)
        h = self.dropout(self.emb(ids))
        if self.use_lstm:
            h = self.dropout(run_lstm(self.lstm, h, lengths))

        size = ids.shape[1]
        eye = torch.eye(size, device=ids.device).unsqueeze(0)
        self_loops = eye * real.unsqueeze(2).float()
        a = adj + self_loops
        degree = a.sum(dim=2, keepdim=True).clamp(min=1)

        for k in range(len(self.layers)):
            h = torch.relu(self.layers[k](torch.bmm(a, h) / degree))
            if k < len(self.layers) - 1:
                h = self.dropout(h)

        return pool_and_classify(h, real, e1, e2, self.dropout, self.out)
