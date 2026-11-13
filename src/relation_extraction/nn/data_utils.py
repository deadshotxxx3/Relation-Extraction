"""Подготовка примеров для нейросетевых моделей.

Здесь только обычный Python, без PyTorch: чтение jsonl, выбор слов
(всё предложение или кратчайший путь), словарь, кодирование слов в номера,
и маски сущностей.
"""
import json
from collections import Counter

PAD, UNK = "<pad>", "<unk>"


def load_jsonl(path):
    """Читает файл, где каждая строка — один пример в формате JSON.
    Возвращает список словарей."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def select_positions(ex, mode):
    """Номера слов, которые пойдут в модель.
    mode == "sentence": все слова предложения.
    mode == "sdp": слова кратчайшего пути, а если его нет - слова обеих сущностей.
    Если не нашлись и сущности, берётся всё предложение, чтобы вход не был пустым."""
    if mode == "sentence":
        return list(range(len(ex["tokens"])))
    if mode != "sdp":
        raise ValueError("mode must be 'sentence' or 'sdp', got " + repr(mode))
    if ex["sdp"] is not None:
        return list(ex["sdp"])
    positions = sorted(ex["e1"] + ex["e2"])
    if len(positions) == 0:
        return list(range(len(ex["tokens"])))
    return positions


def build_vocab(examples):
    """Словарь слово -> номер по всем tokens, слова в нижнем регистре.
    <pad> = 0, <unk> = 1, остальные — от частых к редким."""
    counts = Counter(word.lower() for ex in examples for word in ex["tokens"])
    vocab = {PAD: 0, UNK: 1}
    for word, count in counts.most_common():
        vocab[word] = len(vocab)
    return vocab


def encode(ex, vocab, mode):
    """Номера слов для позиций select_positions(ex, mode).
    Слово, которого нет в словаре, -> 1."""
    positions = select_positions(ex, mode)
    return [vocab.get(ex["tokens"][i].lower(), 1) for i in positions]


def entity_masks(ex, positions):
    """Для каждой выбранной позиции: стоит ли там первая сущность и стоит ли вторая."""
    e1 = [i in ex["e1"] for i in positions]
    e2 = [i in ex["e2"] for i in positions]
    return e1, e2
