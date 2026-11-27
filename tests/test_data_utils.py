import pytest

from src.relation_extraction.nn.data_utils import (
    load_jsonl,
    select_positions,
    build_vocab,
    encode,
    entity_masks,
    build_adjacency,
    load_glove,
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


class TestBuildAdjacency:
    """build_adjacency: матрица смежности дерева для GCN."""

    def test_sentence_mode_is_whole_tree(self, rows):
        ex = rows[0]
        adj = build_adjacency(ex, select_positions(ex, "sentence"))
        edges = sum(sum(row) for row in adj)
        assert edges == 2 * (len(ex["tokens"]) - 1)

    def test_symmetric_without_self_loops(self, rows):
        ex = rows[1]
        adj = build_adjacency(ex, select_positions(ex, "sentence"))
        for a in range(len(adj)):
            assert adj[a][a] == 0
            for b in range(len(adj)):
                assert adj[a][b] == adj[b][a]

    def test_edge_to_head(self, rows):
        ex = rows[0]
        adj = build_adjacency(ex, select_positions(ex, "sentence"))
        assert adj[1][5] == 1
        assert adj[1][0] == 1
        assert adj[1][2] == 0

    def test_sdp_mode_is_chain(self, rows):
        ex = rows[0]
        adj = build_adjacency(ex, select_positions(ex, "sdp"))
        assert adj == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

    def test_every_path_is_chain(self, rows):
        for ex in rows:
            if ex["sdp"] is None:
                continue
            adj = build_adjacency(ex, ex["sdp"])
            for k in range(len(adj) - 1):
                assert adj[k][k + 1] == 1


class TestLoadGlove:
    """load_glove: матрица эмбеддингов из файла GloVe."""

    def test_known_unknown_and_pad(self, tmp_path):
        glove = tmp_path / "glove.txt"
        glove.write_text("the 1 2 3\nzebra 4 5 6\ncat 7 8 9\n", encoding="utf-8")
        vocab = {"<pad>": 0, "<unk>": 1, "the": 2, "cat": 3, "dog": 4}
        matrix, found = load_glove(str(glove), vocab, 3)
        assert found == 2
        assert len(matrix) == 5
        assert matrix[0] == [0.0, 0.0, 0.0]
        assert matrix[2] == [1.0, 2.0, 3.0]
        assert matrix[3] == [7.0, 8.0, 9.0]
        assert len(matrix[4]) == 3

    def test_wrong_dimension(self, tmp_path):
        glove = tmp_path / "glove.txt"
        glove.write_text("the 1 2\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_glove(str(glove), {"<pad>": 0, "<unk>": 1, "the": 2}, 3)
