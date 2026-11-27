import torch
from torch.utils.data import Dataset

from src.relation_extraction.nn.data_utils import (
    load_jsonl,
    select_positions,
    encode,
    entity_masks,
    build_adjacency,
)


class REDataset(Dataset):
    """Примеры одного файла jsonl, уже закодированные словарём.

    mode = "sentence" — модель видит всё предложение,
    mode = "sdp" — только кратчайший путь между сущностями."""

    def __init__(self, path, vocab, mode):
        self.examples = load_jsonl(path)
        self.vocab = vocab
        self.mode = mode
        self.items = []
        for ex in self.examples:
            positions = select_positions(ex, mode)
            ids = encode(ex, vocab, mode)
            e1, e2 = entity_masks(ex, positions)
            adj = build_adjacency(ex, positions)
            self.items.append((ids, e1, e2, ex["label_id"], adj))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def ids(self):
        return [ex["id"] for ex in self.examples]

    def gold_labels(self):
        return [ex["label"] for ex in self.examples]


def collate(batch):
    """Собирает примеры в пачку и дополняет короткие до общей длины.
    batch = [(ids, e1, e2, label, ...), ...]"""
    max_len = max(len(item[0]) for item in batch)

    def pad(seq, value):
        return list(seq) + [value] * (max_len - len(seq))

    ids = torch.tensor([pad(b[0], 0) for b in batch])
    e1 = torch.tensor([pad(b[1], False) for b in batch])
    e2 = torch.tensor([pad(b[2], False) for b in batch])
    lengths = torch.tensor([len(b[0]) for b in batch])
    labels = torch.tensor([b[3] for b in batch])
    return ids, lengths, e1, e2, labels


def collate_gcn(batch):
    """То же, что collate, и в конце ещё матрицы смежности формы [B, T, T].
    batch = [(ids, e1, e2, label, adj), ...]"""
    ids, lengths, e1, e2, labels = collate(batch)
    max_len = ids.shape[1]
    adj = torch.zeros(len(batch), max_len, max_len)
    for k in range(len(batch)):
        matrix = batch[k][4]
        size = len(matrix)
        adj[k, :size, :size] = torch.tensor(matrix, dtype=torch.float)
    return ids, lengths, e1, e2, labels, adj
