import pytest

from src.relation_extraction.nn.data_utils import (
    load_jsonl,
    select_positions,
    build_vocab,
    encode,
    entity_masks,
)

SAMPLE_PATH = "tests/sample.jsonl"


@pytest.fixture
def rows():
    """Все 20 примеров из sample.jsonl — общие данные для тестов."""
    return load_jsonl(SAMPLE_PATH)


class TestLoadJsonl:
    """load_jsonl: чтение файла с примерами."""

    def test_reads_all_examples(self, rows):
        assert len(rows) == 20

    def test_first_example_id(self, rows):
        assert rows[0]["id"] == 1


class TestSelectPositions:
    """select_positions: какие слова предложения пойдут в модель."""

    def test_sentence_mode_takes_all_words(self, rows):
        assert select_positions(rows[0], "sentence") == list(range(17))

    def test_sdp_mode_takes_path(self, rows):
        assert select_positions(rows[0], "sdp") == [12, 13, 15]

    def test_sdp_positions_point_to_path_words(self, rows):
        ex = rows[0]
        words = [ex["tokens"][i] for i in select_positions(ex, "sdp")]
        assert words == ["configuration", "of", "elements"]

    def test_no_path_falls_back_to_entities(self):
        ex = {"tokens": ["a", "b", "c", "d"], "e1": [3], "e2": [0, 1], "sdp": None}
        assert select_positions(ex, "sdp") == [0, 1, 3]

    def test_no_path_and_no_entities_falls_back_to_sentence(self):
        ex = {"tokens": ["a", "b", "c"], "e1": [], "e2": [], "sdp": None}
        assert select_positions(ex, "sdp") == [0, 1, 2]

    def test_unknown_mode(self, rows):
        with pytest.raises(ValueError):
            select_positions(rows[0], "path")


class TestBuildVocab:
    """build_vocab: словарь слово -> номер."""

    def test_special_tokens(self, rows):
        vocab = build_vocab(rows)
        assert vocab["<pad>"] == 0
        assert vocab["<unk>"] == 1

    def test_most_frequent_word_first(self, rows):
        assert build_vocab(rows)["the"] == 2

    def test_size(self, rows):
        assert len(build_vocab(rows)) == 210


class TestEncode:
    """encode: слова -> номера."""

    def test_sdp_words_roundtrip(self, rows):
        vocab = build_vocab(rows)
        ids = encode(rows[0], vocab, "sdp")
        id_to_word = {i: w for w, i in vocab.items()}
        assert [id_to_word[i] for i in ids] == ["configuration", "of", "elements"]

    def test_unknown_word(self, rows):
        vocab = build_vocab(rows)
        ex = {"tokens": ["spaceship"], "e1": [0], "e2": [0], "sdp": [0]}
        assert encode(ex, vocab, "sdp") == [1]


class TestEntityMasks:
    """entity_masks: где стоят сущности."""

    def test_sdp_masks(self, rows):
        ex = rows[0]
        e1, e2 = entity_masks(ex, select_positions(ex, "sdp"))
        assert e1 == [True, False, False]
        assert e2 == [False, False, True]

    def test_multiword_entity(self, rows):
        ex = next(r for r in rows if r["id"] == 9)
        e1, e2 = entity_masks(ex, select_positions(ex, "sentence"))
        assert sum(e1) + sum(e2) == len(ex["e1"]) + len(ex["e2"])
        assert max(sum(e1), sum(e2)) == 2
