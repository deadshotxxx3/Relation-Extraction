import json
import random
from collections import Counter

PAD, UNK = "<pad>", "<unk>"


def load_jsonl(path):
    """Читает файл, где каждая строка - один пример в формате JSON.
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


def build_adjacency(ex, positions):
    """Матрица смежности дерева зависимостей для выбранных позиций.

    adj[a][b] = 1, если слова positions[a] и positions[b] связаны в дереве
    (одно из них - главное слово другого). Связи в обе стороны, без петель:
    петли добавляет сама модель GCN. Для режима sdp получается цепочка из слов пути."""
    place = {}
    for k in range(len(positions)):
        place[positions[k]] = k

    size = len(positions)
    adj = [[0] * size for _ in range(size)]
    for k in range(size):
        head = ex["heads"][positions[k]]
        if head == -1 or head not in place:
            continue
        other = place[head]
        if other == k:
            continue
        adj[k][other] = 1
        adj[other][k] = 1
    return adj


def load_glove(path, vocab, dim, seed=42):
    """Матрица эмбеддингов для словаря из файла GloVe (например glove.6B.100d.txt).

    Строка i - вектор слова с номером i, строка <pad> - нули.
    Слова, которых нет в GloVe, получают маленькие случайные числа.
    Возвращает (матрица как список списков, сколько слов нашлось в GloVe)."""
    rng = random.Random(seed)
    matrix = []
    for i in range(len(vocab)):
        matrix.append([rng.uniform(-0.1, 0.1) for _ in range(dim)])
    matrix[vocab[PAD]] = [0.0] * dim

    found = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word not in vocab or word == PAD or word == UNK:
                continue
            values = parts[1:]
            if len(values) != dim:
                raise ValueError(
                    "GloVe vector for " + repr(word) + " has " + str(len(values))
                    + " numbers, expected " + str(dim)
                )
            matrix[vocab[word]] = [float(v) for v in values]
            found += 1
    return matrix, found
