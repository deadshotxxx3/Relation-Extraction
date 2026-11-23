from pathlib import Path
import pytest
import json

from scripts.compare_errors import (
    load_preds,
    compare,
    top_confusions,
    write_examples
)


DATA_DIR = Path(__file__).parent / "data"


def test_load_preds():
    path = DATA_DIR / "model_a_test_preds.jsonl"

    predictions = load_preds(path)

    assert len(predictions) == 5
    assert predictions[1] == (
        "Cause-Effect(e1,e2)",
        "Cause-Effect(e1,e2)"
    )


def test_load_preds_duplicate_id(tmp_path):
    path = tmp_path / "duplicate_preds.jsonl"

    path.write_text(
        '{"id": 1, "gold": "Other", "pred": "Other"}\n'
        '{"id": 1, "gold": "Other", "pred": "Other"}\n',
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Duplicate example id"):
        load_preds(path)


def test_compare_all_groups():
    preds_a = load_preds(DATA_DIR / "model_a_test_preds.jsonl")
    preds_b = load_preds(DATA_DIR / "model_b_test_preds.jsonl")

    both_right, only_a, only_b, both_wrong = compare(preds_a, preds_b)

    assert both_right == [1]
    assert only_a == [2]
    assert only_b == [3]
    assert both_wrong == [4, 5]


def test_compare_different_ids():
    preds_a = {
        1: ("Other", "Other"),
        2: ("Other", "Other"),
    }

    preds_b = {
        1: ("Other", "Other"),
        3: ("Other", "Other"),
    }

    with pytest.raises(ValueError, match="same example IDs"):
        compare(preds_a, preds_b)


def test_compare_different_gold():
    preds_a = {
        1: ("Other", "Other"),
    }

    preds_b = {
        1: ("Cause-Effect(e1,e2)", "Other"),
    }

    with pytest.raises(ValueError, match="Different gold labels"):
        compare(preds_a, preds_b)


def test_top_confusions():
    preds = load_preds(DATA_DIR / "model_a_test_preds.jsonl")

    result = top_confusions(preds)

    assert result == [
        ("Component-Whole(e1,e2)", "Other", 2),
        ("Member-Collection(e1,e2)", "Other", 1),
    ]


def test_top_confusions_limit():
    preds = load_preds(DATA_DIR / "model_a_test_preds.jsonl")

    result = top_confusions(preds, k=1)

    assert result == [
        ("Component-Whole(e1,e2)", "Other", 2),
    ]


def test_write_examples(tmp_path):
    data_path = tmp_path / "examples.jsonl"
    output_path = tmp_path / "reports" / "errors.md"

    examples = [
        {
            "id": 1,
            "tokens": ["The", "configuration", "of", "antenna", "elements", "."],
            "e1": [1],
            "e2": [4],
            "label": "Component-Whole(e2,e1)",
            "sdp": [1, 2, 4],
            "sdp_deps": ["prep<-", "pobj<-"],
        },
        {
            "id": 2,
            "tokens": ["The", "crucible", "contains", "lawsonite", "."],
            "e1": [3],
            "e2": [1],
            "label": "Instrument-Agency(e2,e1)",
            "sdp": [3, 2, 1],
            "sdp_deps": ["nsubj<-", "dobj<-"],
        },
    ]

    data_path.write_text(
        "".join(
            json.dumps(example) + "\n"
            for example in examples
        ),
        encoding="utf-8",
    )

    preds_a = {
        1: ("Component-Whole(e2,e1)", "Component-Whole(e2,e1)"),
        2: ("Instrument-Agency(e2,e1)", "Other"),
    }

    preds_b = {
        1: ("Component-Whole(e2,e1)", "Other"),
        2: ("Instrument-Agency(e2,e1)", "Instrument-Agency(e2,e1)"),
    }

    write_examples(
        ([1], [2]),
        data_path,
        preds_a,
        preds_b,
        "model_a",
        "model_b",
        output_path,
    )

    content = output_path.read_text(encoding="utf-8")

    assert "Правильно ответила только model_a" in content
    assert "Правильно ответила только model_b" in content

    assert "Пример 1" in content
    assert "Пример 2" in content

    assert "[configuration]e1" in content
    assert "[elements]e2" in content

    assert "prep<-" in content
    assert "pobj<-" in content

    assert "Component-Whole(e2,e1)" in content
    assert "Instrument-Agency(e2,e1)" in content