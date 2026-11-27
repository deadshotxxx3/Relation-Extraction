import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from scripts.train_nn import main

SAMPLE_PATH = str(Path(__file__).parent / "sample.jsonl")


def read_f1(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["official_f1"]


class TestTrainNnScript:
    """scripts/train_nn.py: обучение, сохранение модели и оценка готовой модели."""

    @pytest.mark.parametrize("model", ["bilstm", "gcn"])
    def test_train_save_and_load(self, tmp_path, monkeypatch, model):
        monkeypatch.chdir(tmp_path)

        main([
            "--model", model, "--input", "sdp",
            "--train", SAMPLE_PATH, "--test", SAMPLE_PATH,
            "--epochs", "2", "--patience", "0",
        ])

        name = model + "_sdp"
        assert (tmp_path / "models" / (name + ".pt")).exists()
        trained_f1 = read_f1(tmp_path / "results" / (name + "_sample.json"))

        main([
            "--checkpoint", str(tmp_path / "models" / (name + ".pt")),
            "--name", name + "_loaded", "--test", SAMPLE_PATH,
        ])

        loaded_f1 = read_f1(tmp_path / "results" / (name + "_loaded_sample.json"))
        assert loaded_f1 == pytest.approx(trained_f1)

    def test_glove_file(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        glove = tmp_path / "glove.txt"
        glove.write_text("the " + " ".join(["0.5"] * 8) + "\n", encoding="utf-8")

        main([
            "--train", SAMPLE_PATH, "--glove", str(glove), "--emb-dim", "8",
            "--epochs", "1", "--patience", "0",
        ])

        assert "GloVe vectors found for 1 of" in capsys.readouterr().out
        assert (tmp_path / "models" / "bilstm_sentence.pt").exists()
