import json
from pathlib import Path
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from src.relation_extraction.config import SEED
from src.relation_extraction.baseline import (
    make_text,
    load_split,
    train_baseline,
    predict_baseline,
)

from src.relation_extraction.evaluate import evaluate, official_f1
import shutil
from scripts.run_baseline import main


@pytest.fixture
def sample_path():
    return Path(__file__).parent / "sample.jsonl"


@pytest.fixture
def example():
    return {
        "id": 1,
        "tokens": ["The", "system", "uses", "a", "tool", "."],
        "e1": [1],
        "e2": [4],
        "label": "Instrument-Agency(e2,e1)",
        "sdp": [1, 2, 4],
        "sdp_deps": ["nsubj->", "dobj<-"],
    }


class TestMakeText:

    def test_sentence_mode(self, example):
        result = make_text(example, "sentence")

        assert result == "the system uses a tool ."

    def test_sdp_mode(self, example):
        result = make_text(example, "sdp")

        assert result == "system uses tool"

    def test_sdp_deps_mode(self, example):
        result = make_text(example, "sdp_deps")

        assert result == "system nsubj-> uses dobj<- tool"

    def test_sdp_without_path(self, example):
        example["sdp"] = None
        example["sdp_deps"] = None

        result = make_text(example, "sdp")

        assert result == "system tool"

    def test_sdp_deps_without_path(self, example):
        example["sdp"] = None
        example["sdp_deps"] = None

        result = make_text(example, "sdp_deps")

        assert result == "system tool"

    def test_unknown_mode_raises_error(self, example):
        with pytest.raises(ValueError, match="Unknown text mode"):
            make_text(example, "unknown")

    def test_invalid_sdp_deps_length_raises_error(self, example):
        example["sdp_deps"] = ["nsubj->"]

        with pytest.raises(ValueError, match="dependency relations"):
            make_text(example, "sdp_deps")

    def test_first_sample_sentence(self, sample_path):
        with sample_path.open("r", encoding="utf-8") as file:
            first_example = json.loads(next(file))

        result = make_text(first_example, "sentence")

        assert result == ("the system as described above has its greatest application "
                          "in an arrayed configuration of antenna elements .")

class TestLoadSplit:

    def test_load_split(self, tmp_path, example):
        path = tmp_path / "sample.jsonl"

        second_example = example.copy()
        second_example["id"] = 2
        second_example["tokens"] = ["A", "robot", "moves", "."]

        with path.open("w", encoding="utf-8") as file:
            file.write(json.dumps(example) + "\n")
            file.write("\n")
            file.write(json.dumps(second_example) + "\n")

        ids, texts, labels = load_split(path, "sentence")

        assert ids == [1, 2]

        assert texts == [
            "the system uses a tool .",
            "a robot moves .",
        ]

        assert labels == [
            "Instrument-Agency(e2,e1)",
            "Instrument-Agency(e2,e1)",
        ]


class TestTrainBaseline:

    def test_train_baseline(self):
        texts = [
            "red apple",
            "green apple",
            "blue car",
            "black car",
        ]

        labels = [
            "Cause-Effect(e1,e2)",
            "Cause-Effect(e1,e2)",
            "Other",
            "Other",
        ]

        model = train_baseline(texts, labels)

        assert isinstance(model.named_steps["tfidf"], TfidfVectorizer)
        assert isinstance(model.named_steps["clf"], LogisticRegression)

    def test_vectorizer_settings(self):
        texts = [
            "red apple",
            "green apple",
            "blue car",
            "black car",
        ]

        labels = [
            "Cause-Effect(e1,e2)",
            "Cause-Effect(e1,e2)",
            "Other",
            "Other",
        ]

        model = train_baseline(texts, labels)

        vectorizer = model.named_steps["tfidf"]

        assert vectorizer.token_pattern == r"\S+"
        assert vectorizer.ngram_range == (1, 2)

    def test_classifier_settings(self):
        texts = [
            "red apple",
            "green apple",
            "blue car",
            "black car",
        ]

        labels = [
            "Cause-Effect(e1,e2)",
            "Cause-Effect(e1,e2)",
            "Other",
            "Other",
        ]

        model = train_baseline(texts, labels)

        classifier = model.named_steps["clf"]

        assert classifier.C == 1.0
        assert classifier.max_iter == 1000
        assert classifier.random_state == SEED

    def test_different_lengths_raise_error(self):
        texts = ["red apple", "blue car"]
        labels = ["Other"]

        with pytest.raises(ValueError, match="same length"):
            train_baseline(texts, labels)

    def test_unknown_label_raises_error(self):
        texts = ["red apple", "blue car"]
        labels = ["Unknown-Relation", "Other"]

        with pytest.raises(ValueError, match="Unknown labels"):
            train_baseline(texts, labels)

    def test_dependency_direction_is_kept_in_vocabulary(self, sample_path):
        _, texts, labels = load_split(sample_path, "sdp_deps")
        model = train_baseline(texts, labels, C=10)
        vocabulary = model.named_steps["tfidf"].vocabulary_

        assert "prep<-" in vocabulary


class TestPredictBaseline:

    def test_predict_baseline_returns_list(self):
        texts = [
            "red apple",
            "green apple",
            "blue car",
            "black car",
        ]

        labels = [
            "Cause-Effect(e1,e2)",
            "Cause-Effect(e1,e2)",
            "Other",
            "Other",
        ]

        model = train_baseline(texts, labels)

        predictions = predict_baseline(model, texts)

        assert isinstance(predictions, list)
        assert len(predictions) == len(texts)

        for prediction in predictions:
            assert isinstance(prediction, str)

class TestBaselineQuality:

    @pytest.mark.parametrize("mode", ["sentence", "sdp", "sdp_deps"])
    def test_f1_is_100_on_sample_with_c_10(self, sample_path, mode):
        _, texts, labels = load_split(sample_path, mode)

        model = train_baseline(texts, labels, C=10)

        predictions = predict_baseline(model, texts)

        score, _ = official_f1(labels, predictions)

        assert score == pytest.approx(100.0)


class TestRunBaseline:

    def test_run_baseline_script(
        self,
        tmp_path,
        monkeypatch,
        sample_path,
        capsys,
    ):
        monkeypatch.chdir(tmp_path)

        for filename in (
            "train.jsonl",
            "dev.jsonl",
            "test.jsonl",
            "test_noisy.jsonl",
        ):
            shutil.copyfile(sample_path, tmp_path / filename)

        main([
            "--mode", "sentence",
            "--train", "train.jsonl",
            "--dev", "dev.jsonl",
            "--test", "test.jsonl", "test_noisy.jsonl",
            "--C", "0.1", "1", "10",
        ])

        results_dir = tmp_path / "results"

        assert (results_dir / "baseline_sentence_test.json").exists()
        assert (results_dir / "baseline_sentence_test_preds.jsonl").exists()
        assert (results_dir / "baseline_sentence_test_cm.png").exists()

        assert (results_dir / "baseline_sentence_test_noisy.json").exists()
        assert (results_dir / "baseline_sentence_test_noisy_preds.jsonl").exists()
        assert (results_dir / "baseline_sentence_test_noisy_cm.png").exists()

        assert not (results_dir / "baseline_sentence_dev.json").exists()

        with (results_dir / "baseline_sentence_test.json").open(
            "r", encoding="utf-8"
        ) as file:
            result = json.load(file)

        assert result["official_f1"] == pytest.approx(100.0)
        assert result["split"] == "test"

        output = capsys.readouterr().out
        assert "Selected C =" in output