import json
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

from src.relation_extraction.config import LABELS


OTHER_LABEL = "Other"
RELATION_TYPES = tuple(
    dict.fromkeys(
        label.split("(", maxsplit=1)[0]
        for label in LABELS
        if label != OTHER_LABEL
    )
)


def relation_type(label: str) -> str:
    return label.split("(", maxsplit=1)[0]


def _validate_labels(gold: Sequence[str], pred: Sequence[str]) -> None:
    for field_name, labels in (("gold", gold), ("pred", pred)):
        unknown_labels = [
            label for label in labels
            if label not in LABELS
        ]

        if unknown_labels:
            unique_unknown = list(dict.fromkeys(unknown_labels))
            raise ValueError(
                f"Unknown labels in {field_name}: {unique_unknown}"
            )


def _validate_metric_inputs(
    gold: Sequence[str],
    pred: Sequence[str],
) -> None:
    if len(gold) != len(pred):
        raise ValueError(
            f"gold and pred must have the same length: "
            f"{len(gold)} != {len(pred)}"
        )

    _validate_labels(gold, pred)


def official_f1(gold: list[str], pred: list[str]) -> tuple[float, dict]:
    _validate_metric_inputs(gold, pred)

    present_relations = {
        relation_type(label)
        for label in gold
        if label != OTHER_LABEL
    }

    per_relation = {}

    for relation in RELATION_TYPES:
        if relation not in present_relations:
            continue

        correct = sum(
            1
            for true_label, predicted_label in zip(gold, pred)
            if true_label == predicted_label
            and relation_type(true_label) == relation
        )

        predicted = sum(
            1
            for predicted_label in pred
            if relation_type(predicted_label) == relation
        )

        actual = sum(
            1
            for true_label in gold
            if relation_type(true_label) == relation
        )

        precision = correct / predicted if predicted > 0 else 0.0
        recall = correct / actual if actual > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        per_relation[relation] = {
            "p": precision * 100,
            "r": recall * 100,
            "f1": f1 * 100,
        }

    if not per_relation:
        return 0.0, per_relation

    macro_f1 = sum(
        metrics["f1"]
        for metrics in per_relation.values()
    ) / len(per_relation)

    return macro_f1, per_relation


def save_confusion_matrix(
    gold: list[str],
    pred: list[str],
    path: str | Path,
) -> None:
    _validate_metric_inputs(gold, pred)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(
        y_true=gold,
        y_pred=pred,
        labels=LABELS,
    )

    fig, ax = plt.subplots(figsize=(18, 16))

    image = ax.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=range(len(LABELS)),
        yticks=range(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=90,
        ha="center",
        fontsize=8,
    )

    plt.setp(
        ax.get_yticklabels(),
        fontsize=8,
    )

    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            value = matrix[row, column]

            if value > 0:
                ax.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate(
    ids: list[int | str],
    gold: list[str],
    pred: list[str],
    name: str,
    split: str = "test",
) -> dict:

    if not (len(ids) == len(gold) == len(pred)):
        raise ValueError(
            "ids, gold and pred must have the same length: "
            f"ids={len(ids)}, gold={len(gold)}, pred={len(pred)}"
        )

    _validate_labels(gold, pred)
    official_score, per_relation = official_f1(gold, pred)
    accuracy = (
        sum(
            true_label == predicted_label
            for true_label, predicted_label in zip(gold, pred)
        ) / len(gold)
        if gold else 0.0
    )

    if gold:
        report = classification_report(
            y_true=gold,
            y_pred=pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )

        per_label = {
            label: report[label]
            for label in LABELS
        }
    else:
        per_label = {
            label: {
                "precision": 0.0,
                "recall": 0.0,
                "f1-score": 0.0,
                "support": 0,
            }
            for label in LABELS
        }

    result = {
        "name": name,
        "split": split,
        "n": len(ids),
        "official_f1": official_score,
        "accuracy": accuracy,
        "per_relation": per_relation,
        "per_label": per_label,
    }

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = f"{name}_{split}"
    json_path = results_dir / f"{file_prefix}.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=4,
        )

    predictions_path = results_dir / f"{file_prefix}_preds.jsonl"

    with predictions_path.open("w", encoding="utf-8") as file:
        for example_id, true_label, predicted_label in zip(
            ids, gold, pred
        ):
            record = {
                "id": example_id,
                "gold": true_label,
                "pred": predicted_label,
            }

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    confusion_matrix_path = (
        results_dir / f"{file_prefix}_cm.png"
    )

    save_confusion_matrix(
        gold=gold,
        pred=pred,
        path=confusion_matrix_path,
    )

    print(
        f"{name} {split}: "
        f"F1 = {official_score:.2f}, "
        f"acc = {accuracy * 100:.2f}"
    )

    return result


def read_scorer_file(path: str | Path) -> dict[str, str]:
    path = Path(path)
    result = {}

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t", maxsplit=1)

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid scorer format at line {line_number}: "
                    f"expected 'id<TAB>label'"
                )

            example_id = parts[0].strip()
            label = parts[1].strip()

            if not example_id:
                raise ValueError(
                    f"Empty id at line {line_number}"
                )

            if label not in LABELS:
                raise ValueError(
                    f"Unknown label {label!r} "
                    f"at line {line_number}"
                )

            if example_id in result:
                raise ValueError(
                    f"Duplicate id {example_id!r} "
                    f"at line {line_number}"
                )

            result[example_id] = label

    return result
