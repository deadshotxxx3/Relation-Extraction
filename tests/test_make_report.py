import json
from pathlib import Path

import pytest

from scripts.make_report import (
    MODEL_ORDER,
    load_results,
    model_order,
    build_table,
    plot_f1,
    plot_per_relation,
)


@pytest.fixture
def results_dir(tmp_path):
    directory = tmp_path / "results"
    directory.mkdir()

    results = [
        {
            "name": "baseline_sentence",
            "split": "test",
            "official_f1": 80.0,
            "accuracy": 0.8,
            "per_relation": {
                "Cause-Effect": {"p": 80.0, "r": 80.0, "f1": 80.0},
                "Component-Whole": {"p": 70.0, "r": 70.0, "f1": 70.0},
            },
        },
        {
            "name": "baseline_sentence",
            "split": "test_noisy",
            "official_f1": 75.5,
            "accuracy": 0.75,
            "per_relation": {
                "Cause-Effect": {"p": 75.0, "r": 75.0, "f1": 75.0},
            },
        },
        {
            "name": "baseline_sdp",
            "split": "test",
            "official_f1": 68.0,
            "accuracy": 0.7,
            "per_relation": {
                "Cause-Effect": {"p": 68.0, "r": 68.0, "f1": 68.0},
            },
        },
        {
            "name": "other_model",
            "split": "test",
            "official_f1": 60.0,
            "accuracy": 0.65,
            "per_relation": {},
        },
    ]

    for index, result in enumerate(results):
        path = directory / f"result_{index}.json"

        path.write_text(
            json.dumps(result),
            encoding="utf-8",
        )

    (directory / "invalid.json").write_text(
        '{"official_f1": 50.0}',
        encoding="utf-8",
    )

    (directory / "predictions_preds.jsonl").write_text(
        "this is not a JSON result",
        encoding="utf-8",
    )

    return directory


class TestLoadResults:

    def test_loads_json_results(self, results_dir):
        results = load_results(results_dir)

        assert len(results) == 4

        assert ("baseline_sentence", "test") in results
        assert ("baseline_sentence", "test_noisy") in results
        assert ("baseline_sdp", "test") in results

    def test_skips_files_without_name_or_split(self, results_dir):
        results = load_results(results_dir)

        assert ("None", "None") not in results
        assert len(results) == 4


class TestModelOrder:

    def test_known_models_follow_model_order(self):
        names = [
            "bert",
            "baseline_sdp",
            "baseline_sentence",
            "gcn_sentence",
        ]

        result = model_order(names)

        assert result == [
            "baseline_sentence",
            "baseline_sdp",
            "gcn_sentence",
            "bert",
        ]

    def test_unknown_models_are_sorted_alphabetically(self):
        names = ["z_model", "a_model", "m_model"]

        assert model_order(names) == [
            "a_model",
            "m_model",
            "z_model",
        ]


class TestBuildTable:

    def test_drop_is_calculated_correctly(self, results_dir):
        results = load_results(results_dir)

        table = build_table(results)

        assert "4.50" in table

    def test_missing_noisy_result_is_displayed_as_dash(self, results_dir):
        results = load_results(results_dir)

        table = build_table(results)

        assert (
            "| baseline_sdp | 68.00 | — | — | 70.00 |"
            in table
        )

    def test_accuracy_is_converted_to_percent(self, results_dir):
        results = load_results(results_dir)

        table = build_table(results)

        assert "80.00" in table
        assert "70.00" in table

    def test_rows_follow_model_order(self, results_dir):
        results = load_results(results_dir)

        table = build_table(results)
        rows = table.splitlines()[2:]

        names = [
            row.split("|")[1].strip()
            for row in rows
        ]

        expected = [
            "baseline_sentence",
            "baseline_sdp",
            "other_model",
        ]

        assert names == expected


class TestPlots:

    def test_plot_f1_creates_png(self, results_dir, tmp_path):
        results = load_results(results_dir)
        path = tmp_path / "f1_by_model.png"

        plot_f1(results, path)

        assert path.is_file()
        assert path.stat().st_size > 0

    def test_plot_per_relation_creates_png(
        self, results_dir, tmp_path
    ):
        results = load_results(results_dir)
        path = tmp_path / "f1_by_relation.png"

        plot_per_relation(
            results,
            names=["baseline_sentence", "baseline_sdp"],
            split="test",
            path=path,
        )

        assert path.is_file()
        assert path.stat().st_size > 0
