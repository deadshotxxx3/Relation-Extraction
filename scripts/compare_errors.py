import json
from pathlib import Path
from collections import Counter


def load_preds(path: str | Path) -> dict[int | str, tuple[str, str]]:
    """
    Загружает предсказания модели из JSONL-файла.

    Returns:
        Словарь {id: (gold, pred)}.
    """

    predictions = {}

    with Path(path).open("r", encoding = "utf-8") as file:
        for line_number, line in enumerate(file, start = 1):
            if not line.strip():
                continue

            record = json.loads(line)
            example_id = record["id"]
            gold = record["gold"]
            pred = record["pred"]

            if example_id in predictions:
                raise ValueError(f"Duplicate example id {example_id!r} at line {line_number}")

            predictions[example_id] = (gold, pred)

    return predictions


def compare(preds_a: dict[int | str, tuple[str, str]], 
            preds_b: dict[int | str, tuple[str, str]]) -> tuple[list, list, list, list]:
    """
    Сравнивает предсказания двух моделей на одинаковых примерах.

    Returns:
        Кортеж из четырёх списков ID:
        both_right, only_a, only_b, both_wrong.
    """

    if preds_a.keys() != preds_b.keys():
       raise ValueError("Models must have predictions for the same example IDs") 

    both_right = []
    only_a = []
    only_b = []
    both_wrong = []

    for example_id, (gold_a, pred_a) in preds_a.items():
        gold_b, pred_b = preds_b[example_id]

        if gold_a != gold_b:
            raise ValueError(f"Different gold labels for example {example_id}")

        a_correct = gold_a == pred_a
        b_correct = gold_b == pred_b

        if a_correct and b_correct:
            both_right.append(example_id)

        elif a_correct and not b_correct:
            only_a.append(example_id)

        elif not a_correct and b_correct:
            only_b.append(example_id)

        else:
            both_wrong.append(example_id)

    return both_right, only_a, only_b, both_wrong


def top_confusions(preds: dict[int | str, tuple[str, str]], k: int = 10) -> list[tuple[str, str, int]]:
    """
    Возвращает k наиболее частых ошибок классификации.

    Returns:
        Список кортежей (gold, pred, count),
        отсортированный по убыванию количества ошибок.
    """

    confusion_counts = Counter()

    for gold, pred in preds.values():
        if gold != pred:
            confusion_counts[(gold, pred)] += 1

    return[(gold, pred, count) for (gold, pred), count in confusion_counts.most_common(k)]