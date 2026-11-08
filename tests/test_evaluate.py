import json
from pathlib import Path
import pytest
from src.relation_extraction.config import LABELS
from src.relation_extraction.evaluate import (
    relation_type,
    official_f1,
    save_confusion_matrix,
    evaluate,
    read_scorer_file,
)


def get_one_label_per_relation() -> list[str]:
    labels = []
    seen = set()

    for label in LABELS:
        if label == "Other":
            continue

        relation = relation_type(label)

        if relation not in seen:
            seen.add(relation)
            labels.append(label)

    return labels


def test_relation_type():
    assert relation_type("Cause-Effect(e1,e2)") == "Cause-Effect"
    assert relation_type("Cause-Effect(e2,e1)") == "Cause-Effect"
    assert relation_type("Other") == "Other"


def test_all_predictions_correct():
    gold = get_one_label_per_relation() + ["Other"]
    pred = gold.copy()

    score, per_relation = official_f1(gold, pred)

    assert score == pytest.approx(100.0)
    assert len(per_relation) == 9

    for metrics in per_relation.values():
        assert metrics["p"] == pytest.approx(100.0)
        assert metrics["r"] == pytest.approx(100.0)
        assert metrics["f1"] == pytest.approx(100.0)


def test_all_predictions_other():
    gold = get_one_label_per_relation()
    pred = ["Other"] * len(gold)

    score, per_relation = official_f1(gold, pred)

    assert score == pytest.approx(0.0)

    for metrics in per_relation.values():
        assert metrics["f1"] == pytest.approx(0.0)


def test_wrong_relation_direction_is_error():
    gold = ["Cause-Effect(e1,e2)"]
    pred = ["Cause-Effect(e2,e1)"]

    score, per_relation = official_f1(gold, pred)

    assert score == pytest.approx(0.0)

    metrics = per_relation["Cause-Effect"]

    assert metrics["p"] == pytest.approx(0.0)
    assert metrics["r"] == pytest.approx(0.0)
    assert metrics["f1"] == pytest.approx(0.0)


def test_different_lengths_raise_error():
    gold = ["Other", "Other"]
    pred = ["Other"]

    with pytest.raises(ValueError, match="same length"):
        official_f1(gold, pred)


def test_evaluate_different_lengths_raise_error():
    ids = [1, 2]
    gold = ["Other", "Other"]
    pred = ["Other"]

    with pytest.raises(ValueError, match="same length"):
        evaluate(
            ids=ids,
            gold=gold,
            pred=pred,
            name="test_model",
        )


def test_unknown_label_raises_error():
    ids = [1]
    gold = ["Unknown-Relation"]
    pred = ["Other"]

    with pytest.raises(ValueError, match="Unknown labels"):
        evaluate(
            ids=ids,
            gold=gold,
            pred=pred,
            name="test_model",
        )


def test_empty_input():
    score, per_relation = official_f1([], [])

    assert score == pytest.approx(0.0)
    assert len(per_relation) == 9


def test_evaluate_creates_all_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    gold = get_one_label_per_relation() + ["Other"]
    pred = gold.copy()
    ids = list(range(1, len(gold) + 1))

    result = evaluate(
        ids=ids,
        gold=gold,
        pred=pred,
        name="test_model",
        split="test",
    )

    results_dir = tmp_path / "results"

    json_path = results_dir / "test_model_test.json"
    predictions_path = results_dir / "test_model_test_preds.jsonl"
    matrix_path = results_dir / "test_model_test_cm.png"

    assert json_path.exists()
    assert predictions_path.exists()
    assert matrix_path.exists()

    with json_path.open("r", encoding="utf-8") as file:
        saved_result = json.load(file)

    assert saved_result == result
    assert saved_result["official_f1"] == pytest.approx(100.0)
    assert saved_result["accuracy"] == pytest.approx(1.0)
    assert saved_result["n"] == len(gold)

    with predictions_path.open("r", encoding="utf-8") as file:
        records = [json.loads(line) for line in file]

    assert len(records) == len(ids)
    assert records[0]["id"] == ids[0]
    assert records[0]["gold"] == gold[0]
    assert records[0]["pred"] == pred[0]


def test_save_confusion_matrix(tmp_path):
    gold = get_one_label_per_relation() + ["Other"]
    pred = gold.copy()

    path = tmp_path / "matrix.png"

    save_confusion_matrix(gold, pred, path)

    assert path.exists()
    assert path.stat().st_size > 0


def test_read_scorer_file(tmp_path):
    path = tmp_path / "scorer.txt"

    path.write_text(
        "1\tOther\n"
        "2\tCause-Effect(e1,e2)\n"
        "3\tProduct-Producer(e2,e1)\n",
        encoding="utf-8",
    )

    result = read_scorer_file(path)

    assert result == {
        "1": "Other",
        "2": "Cause-Effect(e1,e2)",
        "3": "Product-Producer(e2,e1)",
    }


def test_read_scorer_file_duplicate_id(tmp_path):
    path = tmp_path / "scorer.txt"

    path.write_text(
        "1\tOther\n"
        "1\tOther\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate id"):
        read_scorer_file(path)


def test_official_scorer_result():
    test_dir = Path(__file__).parent

    answer_path = test_dir / "answer_key1.txt"
    proposed_path = test_dir / "proposed_answer1.txt"

    if not answer_path.exists() or not proposed_path.exists():
        pytest.skip(
            "Official SemEval scorer files have not been provided yet"
        )

    answer = read_scorer_file(answer_path)
    proposed = read_scorer_file(proposed_path)

    ids = list(answer.keys())

    gold = [
        answer[example_id]
        for example_id in ids
    ]

    pred = [
        proposed.get(example_id, "Other")
        for example_id in ids
    ]

    score, _ = official_f1(gold, pred)

    assert score == pytest.approx(64.09, abs=0.01)
