import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.relation_extraction.evaluate import RELATION_TYPES


MODEL_ORDER = [
    "baseline_sentence",
    "baseline_sdp",
    "baseline_sdp_deps",
    "bilstm_sentence",
    "bilstm_sdp",
    "gcn_sentence",
    "bert",
]

DEFAULT_MODELS = [
    "baseline_sentence",
    "baseline_sdp",
    "bilstm_sdp",
]


def load_results(results_dir: str | Path) -> dict:
    """
    Загружает результаты оценки моделей из JSON-файлов.

    Файлы без name или split пропускаются.

    Returns:
        Словарь вида {(name, split): содержимое JSON}.
    """
    results = {}
    results_dir = Path(results_dir)

    for json_path in sorted(results_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as file:
            result = json.load(file)

        name = result.get("name")
        split = result.get("split")

        if not name or not split:
            continue

        results[(name, split)] = result

    return results


def model_order(names) -> list[str]:
    """
    Сортирует модели согласно MODEL_ORDER.
    Остальные модели располагаются по алфавиту.
    """
    names = set(names)
    known_models = [name for name in MODEL_ORDER if name in names]
    other_models = sorted(names - set(MODEL_ORDER))

    return known_models + other_models


def _get_metric(results, name, split, metric):
    """Безопасно получает отдельную метрику."""
    result = results.get((name, split))

    if result is None:
        return None

    return result.get(metric)


def _format_value(value):
    """Форматирует число с двумя знаками после запятой."""
    if value is None:
        return "—"

    return f"{value:.2f}"


def build_table(results: dict) -> str:
    """
    Создаёт Markdown-таблицу с результатами всех моделей.

    F1 уже хранится в процентах.
    Accuracy переводится из доли в проценты.
    """
    names = model_order(name for name, split in results)
    headers = [
        "Модель",
        "F1 test",
        "F1 test_noisy",
        "Падение",
        "Accuracy test",
    ]

    rows = []

    for name in names:
        test_f1 = _get_metric(results, name, "test", "official_f1")
        noisy_f1 = _get_metric(results, name, "test_noisy", "official_f1")
        accuracy = _get_metric(results, name, "test", "accuracy")

        if test_f1 is not None and noisy_f1 is not None:
            drop = test_f1 - noisy_f1
        else:
            drop = None

        if accuracy is not None:
            accuracy *= 100

        rows.append([name, _format_value(test_f1), _format_value(noisy_f1),
                     _format_value(drop), _format_value(accuracy)])

    separator = ["---"] * len(headers)

    table_lines = ["| " + " | ".join(headers) + " |",
                   "| " + " | ".join(separator) + " |"]

    for row in rows:
        table_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(table_lines)


def plot_f1(results: dict, path: str | Path) -> None:
    """
    Создаёт столбчатую диаграмму F1 для test и test_noisy.
    """
    names = model_order(name for name, split in results)
    positions = list(range(len(names)))

    test_values = [_get_metric(results, name, "test", "official_f1")
                   for name in names]

    noisy_values = [_get_metric(results, name, "test_noisy", "official_f1")
                    for name in names]

    test_heights = [float("nan") if value is None else value
                    for value in test_values]

    noisy_heights = [float("nan") if value is None else value
                     for value in noisy_values]

    fig, ax = plt.subplots(figsize=(max(9, len(names) * 1.5), 6))
    test_bars = ax.bar([x - 0.2 for x in positions], test_heights,
                       width=0.4, label="test")

    noisy_bars = ax.bar([x + 0.2 for x in positions], noisy_heights,
                        width=0.4, label="test_noisy")

    for bars, values in ((test_bars, test_values), (noisy_bars, noisy_values)):
        for bar, value in zip(bars, values):
            if value is None or not math.isfinite(value):
                continue

            vertical_offset = -14 if value >= 95 else 3
            vertical_alignment = "top" if value >= 95 else "bottom"

            ax.annotate(
                f"{value:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, vertical_offset),
                textcoords="offset points",
                ha="center",
                va=vertical_alignment,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=25, ha="right")

    ax.set_ylabel("macro-F1, %")
    ax.set_ylim(0, 100)
    ax.set_title("Macro-F1 by model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_relation(results: dict, names: list[str], split: str, path: str | Path) -> None:
    """
    Создаёт горизонтальную диаграмму F1 по типам отношений.

    Для каждого типа отношения строится отдельная группа
    столбиков — по одному на каждую модель.
    """
    names = [
        name for name in model_order(names)
        if (name, split) in results
    ]

    relation_positions = list(range(len(RELATION_TYPES)))

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    if names:
        bar_height = 0.8 / len(names)

        for model_index, name in enumerate(names):
            result = results[(name, split)]
            per_relation = result.get("per_relation", {})

            values = [
                per_relation.get(relation, {}).get("f1")
                for relation in RELATION_TYPES
            ]

            heights = [
                float("nan") if value is None else value
                for value in values
            ]

            offset = (
                model_index - (len(names) - 1) / 2
            ) * bar_height

            ax.barh(
                [
                    position + offset
                    for position in relation_positions
                ],
                heights,
                height=bar_height * 0.9,
                label=name,
            )

    ax.set_yticks(relation_positions)
    ax.set_yticklabels(RELATION_TYPES)

    ax.set_xlabel("F1, %")
    ax.set_xlim(0, 100)
    ax.set_title(f"F1 by relation ({split})")

    if names:
        ax.legend()

    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a report from model evaluation results."
    )

    parser.add_argument(
        "--results",
        default="results",
        help="Directory containing evaluation JSON files.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models to include in the per-relation plot.",
    )

    args = parser.parse_args(argv)

    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)

    table = build_table(results)

    print(table)

    summary_path = results_dir / "summary.md"

    summary_path.write_text(
        "# Results summary\n\n" + table + "\n",
        encoding="utf-8",
    )

    plot_f1(
        results,
        results_dir / "f1_by_model.png",
    )

    plot_per_relation(
        results,
        args.models,
        "test",
        results_dir / "f1_by_relation.png",
    )


if __name__ == "__main__":
    main()
