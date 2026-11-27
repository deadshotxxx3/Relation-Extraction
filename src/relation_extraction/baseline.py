import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.relation_extraction.config import LABELS, SEED


def make_text(ex: dict, mode: str) -> str:
    """
    Преобразует пример в текстовое представление для обучения модели.

    Поддерживает три режима:
    - sentence: полное предложение;
    - sdp: слова кратчайшего пути между сущностями;
    - sdp_deps: слова пути вместе с зависимостями между ними.

    Если путь отсутствует, вместо него используются слова обеих
    сущностей в порядке их появления в предложении.

    Args:
        ex: Словарь с данными одного примера.
        mode: Режим формирования текста.

    Returns:
        Строка из слов в нижнем регистре, разделённых пробелами.

    Raises:
        ValueError: Если указан неизвестный режим или данные пути
            имеют некорректную структуру.
    """

    if mode not in ("sentence", "sdp", "sdp_deps"):
        raise ValueError(f"Unknown text mode: {mode}")

    tokens = [token.lower() for token in ex["tokens"]]

    if mode == "sentence":
        return " ".join(tokens)

    sdp = ex["sdp"]

    if sdp is None:
        entity_indices = sorted(ex["e1"] + ex["e2"])

        return " ".join(tokens[index] for index in entity_indices)

    sdp_words = [tokens[index] for index in sdp]

    if mode == "sdp":
        return " ".join(sdp_words)

    sdp_deps = ex["sdp_deps"]

    if sdp_deps is None or len(sdp_deps) != len(sdp) - 1:
        raise ValueError("The number of dependency relations must be "
                         "one less than the number of SDP tokens")

    text_parts = []

    for index, word in enumerate(sdp_words):
        text_parts.append(word)

        if index < len(sdp_deps):
            text_parts.append(sdp_deps[index].lower())

    return " ".join(text_parts)


def load_split(path: str | Path, mode: str) -> tuple[list[int | str], list[str], list[str]]:
    """
    Загружает набор данных из JSONL-файла.

    Каждая непустая строка файла преобразуется в текст с помощью
    make_text(). Порядок примеров соответствует порядку в файле.

    Args:
        path: Путь к JSONL-файлу.
        mode: Режим формирования текста.

    Returns:
        Кортеж из трёх списков:
        - идентификаторы примеров;
        - текстовые представления примеров;
        - правильные метки отношений.
    """
    ids = []
    texts = []
    labels = []
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            example = json.loads(line)
            ids.append(example["id"])
            texts.append(make_text(example, mode))
            labels.append(example["label"])

    return ids, texts, labels


def train_baseline(texts: list[str], labels: list[str], C: float = 1.0) -> Pipeline:
    """
    Создаёт и обучает модель классификации отношений.

    Модель представляет собой Pipeline из двух этапов:
    1. TfidfVectorizer преобразует текст в числовые признаки.
    2. LogisticRegression классифицирует полученные признаки.

    Векторизатор использует униграммы и биграммы.
    Параметр token_pattern=r"\\S+" позволяет сохранять
    однобуквенные слова и специальные токены связей.

    Args:
        texts: Текстовые представления обучающих примеров.
        labels: Правильные метки обучающих примеров.
        C: Параметр регуляризации логистической регрессии.

    Returns:
        Обученный Pipeline, готовый к предсказанию меток.
    """
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")

    unknown_labels = list(dict.fromkeys(label for label in labels if label not in LABELS))

    if unknown_labels:
        raise ValueError(f"Unknown labels: {unknown_labels}")

    model = Pipeline([("tfidf", TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2))),
                      ("clf", LogisticRegression(C=C, max_iter=1000, random_state=SEED))])

    model.fit(texts, labels)
    return model

def predict_baseline(model: Pipeline, texts: list[str]) -> list[str]:
    """
    Получает предсказания обученной модели для переданных текстов.

    Args:
        model: Обученный Pipeline.
        texts: Список текстовых представлений примеров.

    Returns:
        Список предсказанных меток отношений.
    """
    return list(model.predict(texts))
