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


@pytest.fixture
def one_label_per_relation() -> list[str]:
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


@pytest.fixture
def evaluation_data(one_label_per_relation: list[str]) -> dict:
    gold = one_label_per_relation + ["Other"]

    return {
        "ids": list(range(1, len(gold) + 1)),
        "gold": gold,
        "pred": gold.copy(),
    }


@pytest.fixture
def evaluation_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evaluation_data: dict,
) -> dict:
    monkeypatch.chdir(tmp_path)

    result = evaluate(
        ids=evaluation_data["ids"],
        gold=evaluation_data["gold"],
        pred=evaluation_data["pred"],
        name="test_model",
        split="test",
    )

    results_dir = tmp_path / "results"

    return {
        "result": result,
        "ids": evaluation_data["ids"],
        "gold": evaluation_data["gold"],
        "pred": evaluation_data["pred"],
        "json_path": results_dir / "test_model_test.json",
        "predictions_path": results_dir / "test_model_test_preds.jsonl",
        "matrix_path": results_dir / "test_model_test_cm.png",
    }


@pytest.fixture
def saved_evaluation_json(evaluation_artifacts: dict) -> dict:
    with evaluation_artifacts["json_path"].open("r", encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture
def saved_prediction_records(evaluation_artifacts: dict) -> list[dict]:
    with evaluation_artifacts["predictions_path"].open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


class TestRelationType:
    def test_extracts_relation_type_with_e1_as_first_argument(self):
        assert relation_type("Cause-Effect(e1,e2)") == "Cause-Effect"

    def test_extracts_relation_type_with_e2_as_first_argument(self):
        assert relation_type("Cause-Effect(e2,e1)") == "Cause-Effect"

    def test_returns_other_for_other_label(self):
        assert relation_type("Other") == "Other"


class TestOfficialF1:
    def test_all_predictions_correct_returns_100(self, one_label_per_relation):
        gold = one_label_per_relation + ["Other"]
        pred = gold.copy()

        score, _ = official_f1(gold, pred)

        assert score == pytest.approx(100.0)

    def test_all_predictions_correct_includes_all_nine_relations(
        self, one_label_per_relation
    ):
        gold = one_label_per_relation + ["Other"]
        pred = gold.copy()

        _, per_relation = official_f1(gold, pred)

        assert len(per_relation) == 9

    @pytest.mark.parametrize("metric_name", ["p", "r", "f1"])
    def test_all_predictions_correct_have_100_per_relation_metric(
        self, one_label_per_relation, metric_name
    ):
        gold = one_label_per_relation + ["Other"]
        pred = gold.copy()

        _, per_relation = official_f1(gold, pred)

        assert all(
            metrics[metric_name] == pytest.approx(100.0)
            for metrics in per_relation.values()
        )

    def test_all_predictions_other_returns_zero(self, one_label_per_relation):
        gold = one_label_per_relation
        pred = ["Other"] * len(gold)

        score, _ = official_f1(gold, pred)

        assert score == pytest.approx(0.0)

    def test_all_predictions_other_have_zero_f1(self, one_label_per_relation):
        gold = one_label_per_relation
        pred = ["Other"] * len(gold)

        _, per_relation = official_f1(gold, pred)

        assert all(
            metrics["f1"] == pytest.approx(0.0)
            for metrics in per_relation.values()
        )

    def test_wrong_relation_direction_returns_zero(self):
        gold = ["Cause-Effect(e1,e2)"]
        pred = ["Cause-Effect(e2,e1)"]

        score, _ = official_f1(gold, pred)

        assert score == pytest.approx(0.0)

    @pytest.mark.parametrize("metric_name", ["p", "r", "f1"])
    def test_wrong_relation_direction_has_zero_metric(self, metric_name):
        gold = ["Cause-Effect(e1,e2)"]
        pred = ["Cause-Effect(e2,e1)"]

        _, per_relation = official_f1(gold, pred)

        assert per_relation["Cause-Effect"][metric_name] == pytest.approx(0.0)

    def test_macro_f1_uses_only_relations_present_in_gold(self):
        gold = ["Cause-Effect(e1,e2)", "Other"]
        pred = gold.copy()

        score, _ = official_f1(gold, pred)

        assert score == pytest.approx(100.0)

    def test_per_relation_contains_only_relations_present_in_gold(self):
        gold = ["Cause-Effect(e1,e2)", "Other"]
        pred = gold.copy()

        _, per_relation = official_f1(gold, pred)

        assert list(per_relation) == ["Cause-Effect"]

    def test_different_lengths_raise_error(self):
        gold = ["Other", "Other"]
        pred = ["Other"]

        with pytest.raises(ValueError, match="same length"):
            official_f1(gold, pred)

    def test_empty_input_returns_zero(self):
        score, _ = official_f1([], [])

        assert score == pytest.approx(0.0)

    def test_empty_input_returns_empty_per_relation(self):
        _, per_relation = official_f1([], [])

        assert per_relation == {}


class TestSaveConfusionMatrix:
    @pytest.fixture
    def matrix_path(self, tmp_path, one_label_per_relation):
        gold = one_label_per_relation + ["Other"]
        path = tmp_path / "matrix.png"

        save_confusion_matrix(gold, gold.copy(), path)

        return path

    def test_creates_matrix_file(self, matrix_path):
        assert matrix_path.exists()

    def test_matrix_file_is_not_empty(self, matrix_path):
        assert matrix_path.stat().st_size > 0


class TestEvaluate:
    def test_different_lengths_raise_error(self):
        with pytest.raises(ValueError, match="same length"):
            evaluate(
                ids=[1, 2],
                gold=["Other", "Other"],
                pred=["Other"],
                name="test_model",
            )

    def test_unknown_label_raises_error(self):
        with pytest.raises(ValueError, match="Unknown labels"):
            evaluate(
                ids=[1],
                gold=["Unknown-Relation"],
                pred=["Other"],
                name="test_model",
            )

    def test_creates_json_file(self, evaluation_artifacts):
        assert evaluation_artifacts["json_path"].is_file()

    def test_creates_predictions_file(self, evaluation_artifacts):
        assert evaluation_artifacts["predictions_path"].is_file()

    def test_creates_confusion_matrix_file(self, evaluation_artifacts):
        assert evaluation_artifacts["matrix_path"].is_file()

    def test_saved_json_matches_returned_result(
        self, evaluation_artifacts, saved_evaluation_json
    ):
        assert saved_evaluation_json == evaluation_artifacts["result"]

    def test_saved_json_contains_official_f1(
        self, saved_evaluation_json
    ):
        assert saved_evaluation_json["official_f1"] == pytest.approx(100.0)

    def test_saved_json_contains_accuracy(self, saved_evaluation_json):
        assert saved_evaluation_json["accuracy"] == pytest.approx(1.0)

    def test_saved_json_contains_example_count(
        self, evaluation_artifacts, saved_evaluation_json
    ):
        assert saved_evaluation_json["n"] == len(evaluation_artifacts["ids"])

    def test_predictions_file_contains_all_examples(
        self, evaluation_artifacts, saved_prediction_records
    ):
        assert len(saved_prediction_records) == len(evaluation_artifacts["ids"])

    def test_predictions_file_saves_first_example_id(
        self, evaluation_artifacts, saved_prediction_records
    ):
        assert saved_prediction_records[0]["id"] == evaluation_artifacts["ids"][0]

    def test_predictions_file_saves_first_gold_label(
        self, evaluation_artifacts, saved_prediction_records
    ):
        assert saved_prediction_records[0]["gold"] == evaluation_artifacts["gold"][0]

    def test_predictions_file_saves_first_predicted_label(
        self, evaluation_artifacts, saved_prediction_records
    ):
        assert saved_prediction_records[0]["pred"] == evaluation_artifacts["pred"][0]


class TestReadScorerFile:
    def test_reads_scorer_file(self, tmp_path):
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

    def test_duplicate_id_raises_error(self, tmp_path):
        path = tmp_path / "scorer.txt"

        path.write_text(
            "1\tOther\n"
            "1\tOther\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Duplicate id"):
            read_scorer_file(path)


class TestOfficialScorerResult:
    def test_official_scorer_result(self):
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
        gold = [answer[example_id] for example_id in ids]
        pred = [proposed.get(example_id, "Other") for example_id in ids]

        score, _ = official_f1(gold, pred)

        assert score == pytest.approx(64.09, abs=0.01)