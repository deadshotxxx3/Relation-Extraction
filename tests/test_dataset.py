import pytest

torch = pytest.importorskip("torch")

from src.relation_extraction.nn.data_utils import build_vocab, load_jsonl
from src.relation_extraction.nn.dataset import REDataset, collate

SAMPLE_PATH = "tests/sample.jsonl"


class TestCollate:
    """collate: сборка примеров разной длины в одну пачку."""

    batch = [
        ([5, 6, 7], [True, False, False], [False, False, True], 3),
        ([8, 9],    [True, False],        [False, True],        0),
    ]

    def test_ids_padded_with_zeros(self):
        ids, lengths, e1, e2, labels = collate(self.batch)
        assert ids.tolist() == [[5, 6, 7], [8, 9, 0]]

    def test_lengths(self):
        ids, lengths, e1, e2, labels = collate(self.batch)
        assert lengths.tolist() == [3, 2]

    def test_masks_padded_with_false(self):
        ids, lengths, e1, e2, labels = collate(self.batch)
        assert e1.tolist() == [[True, False, False], [True, False, False]]
        assert e2.tolist() == [[False, False, True], [False, True, False]]

    def test_labels(self):
        ids, lengths, e1, e2, labels = collate(self.batch)
        assert labels.tolist() == [3, 0]


class TestREDataset:
    """REDataset: примеры sample.jsonl в виде номеров слов и масок."""

    @pytest.fixture
    def vocab(self):
        return build_vocab(load_jsonl(SAMPLE_PATH))

    def test_length(self, vocab):
        assert len(REDataset(SAMPLE_PATH, vocab, "sentence")) == 20

    def test_sdp_item(self, vocab):
        ids, e1, e2, label = REDataset(SAMPLE_PATH, vocab, "sdp")[0]
        assert len(ids) == 3
        assert e1 == [True, False, False]
        assert e2 == [False, False, True]
        assert label == 4

    def test_sentence_item_length(self, vocab):
        ids, e1, e2, label = REDataset(SAMPLE_PATH, vocab, "sentence")[0]
        assert len(ids) == 17

    def test_gold_labels_and_ids(self, vocab):
        dataset = REDataset(SAMPLE_PATH, vocab, "sentence")
        assert dataset.gold_labels()[0] == "Component-Whole(e2,e1)"
        assert dataset.ids()[0] == 1
