import json
from pathlib import Path
from collections import Counter
import re


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


def _format_sentence(example: dict) -> str:
    """
    Восстанавливает предложение и выделяет в нём обе сущности.

    Сущности определяются по индексам токенов e1 и e2.
    """
    tokens = example["tokens"]
    e1_positions = set(example["e1"])
    e2_positions = set(example["e2"])

    if e1_positions & e2_positions:
        raise ValueError("Entity positions must not overlap")

    if (e1_positions | e2_positions) - set(range(len(tokens))):
        raise ValueError("Entity position is outside the sentence")

    parts = []
    index = 0

    while index < len(tokens):
        if index in e1_positions:
            entity_tokens = []

            while index < len(tokens) and index in e1_positions:
                entity_tokens.append(tokens[index])
                index += 1

            entity_text = " ".join(entity_tokens)
            parts.append(f"**[{entity_text}]e1**")

        elif index in e2_positions:
            entity_tokens = []

            while index < len(tokens) and index in e2_positions:
                entity_tokens.append(tokens[index])
                index += 1

            entity_text = " ".join(entity_tokens)
            parts.append(f"**[{entity_text}]e2**")

        else:
            parts.append(tokens[index])
            index += 1

    sentence = " ".join(parts)

    sentence = re.sub(r"\s+([.,!?;:%])", r"\1", sentence)
    sentence = re.sub(r"\s+(['’][A-Za-z]+)", r"\1", sentence)
    sentence = re.sub(r"\s+([-–—])\s+", r"\1", sentence)

    return sentence


def _format_sdp_path(example: dict) -> str:
    """
    Форматирует кратчайший путь между сущностями
    вместе с типами зависимостей.
    """

    tokens = example["tokens"]
    sdp = example.get("sdp")
    sdp_deps = example.get("sdp_deps")

    if not sdp or sdp_deps is None:
        return "Пути нет"

    if len(sdp_deps) != len(sdp) - 1:
        raise ValueError("Invalid SDP dependencies length")

    if any(index < 0 or index >= len(tokens) for index in sdp):
        raise ValueError("SDP token index is outside the sentence")

    parts = [tokens[sdp[0]]]

    for dependency, token_index in zip(sdp_deps, sdp[1:]):
        parts.append(f"—[{dependency}]—")
        parts.append(tokens[token_index])

    return " ".join(parts)


def write_examples(ids: tuple[list[int | str], list[int | str]], data_path: str | Path, preds_a: dict[int | str, tuple[str, str]], 
                   preds_b: dict[int | str, tuple[str, str]], name_a: str, name_b: str, path: str | Path,
                   limit: int = 10) -> None:
    """
    Записывает примеры, на которых одна модель оказалась
    правильной, а другая ошиблась, в Markdown-файл.
    """

    if limit < 0:
        raise ValueError("Limit must not be negative")

    examples = {}

    with Path(data_path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            example = json.loads(line)
            example_id = example["id"]

            if example_id in examples:
                raise ValueError(f"Duplicate example id {example_id!r} at line {line_number}")

            examples[example_id] = example

    lines = [f"# Анализ ошибок: {name_a} vs {name_b}", ""]

    sections = [
        (f"Правильно ответила только {name_a}", ids[0]),
        (f"Правильно ответила только {name_b}", ids[1])
    ]

    for section_title, section_ids in sections:
        lines.extend([f"## {section_title}", ""])

        selected_ids = section_ids[:limit]

        if not selected_ids:
            lines.extend(["Нет примеров.", ""])
            continue

        for example_id in selected_ids:
            if example_id not in examples:
                raise ValueError(f"Example {example_id} not found in data")

            if example_id not in preds_a or example_id not in preds_b:
                raise ValueError(f"Missing predictions for example {example_id}")

            example = examples[example_id]
            gold_a, pred_a = preds_a[example_id]
            gold_b, pred_b = preds_b[example_id]

            if gold_a != gold_b:
                raise ValueError(f"Different gold labels for example {example_id}")

            sentence = _format_sentence(example)
            sdp_path = _format_sdp_path(example)

            lines.extend([f"### Пример {example_id}", "", f"**Предложение:** {sentence}", "",
                          f"**SDP:** {sdp_path}", "", f"**Правильная метка:** `{gold_a}`", "",
                          f"**{name_a}:** `{pred_a}`", "", f"**{name_b}:** `{pred_b}`", ""])

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")