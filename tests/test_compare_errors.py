from pathlib import Path

import pytest

from scripts.compare_errors import (
    load_preds,
    compare,
    top_confusions,
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