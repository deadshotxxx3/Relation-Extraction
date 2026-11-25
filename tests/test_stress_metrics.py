from scripts.stress_test import get_path_signature, path_hits_insertion, word_signature


class TestWordSignature:
    def test_same_words_different_edges_and_positions(self):
        before = {"tokens": ["Rain", "flood"], "sdp": [0, 1], "sdp_deps": ["dep->"]}
        after = {"tokens": ["RAIN", "new", "flood"], "sdp": [0, 2], "sdp_deps": ["obj<-"]}
        assert word_signature(before) == word_signature(after) == ("rain", "flood")
        assert get_path_signature(before) != get_path_signature(after)
        assert word_signature(dict(after, sdp=[2, 0])) == ("flood", "rain")

    def test_missing_path(self):
        assert word_signature({"sdp": None}) is None


class TestInsertionHit:
    def test_repeated_word_and_punctuation(self):
        # "report, report, ends" — вставка ", report," занимает [6, 15).
        offsets = [0, 6, 8, 14, 16]
        record = {"tokens": ["report", ",", "report", ",", "ends"], "sdp": [0, 1, 3, 4]}
        assert not path_hits_insertion(record, offsets, 6, 15)
        assert path_hits_insertion(dict(record, sdp=[0, 2, 4]), offsets, 6, 15)

    def test_half_open_boundaries(self):
        record = {"tokens": ["a", "b"], "sdp": [1]}
        assert not path_hits_insertion(record, [0, 2], 0, 2)
        assert path_hits_insertion(record, [0, 2], 2, 3)