import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader

from src.relation_extraction.config import LABELS
from src.relation_extraction.nn.bilstm import BiLSTMClassifier, masked_max
from src.relation_extraction.nn.data_utils import build_vocab, load_jsonl
from src.relation_extraction.nn.dataset import REDataset, collate
from src.relation_extraction.nn.train import accuracy, predict, train

SAMPLE_PATH = "tests/sample.jsonl"


@pytest.fixture
def vocab():
    return build_vocab(load_jsonl(SAMPLE_PATH))


def first_batch(vocab, mode):
    dataset = REDataset(SAMPLE_PATH, vocab, mode)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate)
    return next(iter(loader))


class TestMaskedMax:
    """masked_max: максимум только по отмеченным словам."""

    def test_ignores_masked_positions(self):
        h = torch.tensor([[[1.0], [5.0], [3.0]]])
        mask = torch.tensor([[True, False, True]])
        assert masked_max(h, mask).tolist() == [[3.0]]


class TestBiLSTM:
    """BiLSTMClassifier: формы выхода."""

    @pytest.mark.parametrize("mode", ["sentence", "sdp"])
    def test_output_shape(self, vocab, mode):
        torch.manual_seed(0)
        model = BiLSTMClassifier(len(vocab), len(LABELS), emb_dim=16, hidden=8)
        ids, lengths, e1, e2, labels = first_batch(vocab, mode)
        assert tuple(model(ids, lengths, e1, e2).shape) == (4, len(LABELS))

    def test_padding_does_not_change_answer(self, vocab):
        torch.manual_seed(0)
        model = BiLSTMClassifier(len(vocab), len(LABELS), emb_dim=16, hidden=8)
        model.eval()
        dataset = REDataset(SAMPLE_PATH, vocab, "sdp")
        alone = collate([dataset[0]])
        together = collate([dataset[0], dataset[1]])
        out_alone = model(alone[0], alone[1], alone[2], alone[3])
        out_together = model(together[0], together[1], together[2], together[3])
        assert torch.allclose(out_alone[0], out_together[0], atol=1e-5)


def overfit(model, vocab, mode):
    dataset = REDataset(SAMPLE_PATH, vocab, mode)
    train_loader = DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dataset, batch_size=20, shuffle=False, collate_fn=collate)
    train(model, train_loader, dev_loader, epochs=150, lr=0.01, weight_decay=0.0, patience=0)
    return accuracy(dataset.gold_labels(), predict(model, dev_loader))


class TestOverfitSample:
    """Главная проверка: BiLSTM может выучить 20 примеров почти на 100%."""

    @pytest.mark.parametrize("mode", ["sentence", "sdp"])
    def test_bilstm(self, vocab, mode):
        torch.manual_seed(0)
        model = BiLSTMClassifier(len(vocab), len(LABELS), emb_dim=32, hidden=32, dropout=0.0)
        assert overfit(model, vocab, mode) >= 0.95
