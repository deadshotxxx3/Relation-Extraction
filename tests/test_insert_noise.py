from dataclasses import replace

import pytest

from scripts.stress_test import NOISE_TEMPLATES, insert_noise
from src.relation_extraction.data.semeval_loader import SemEvalExample


def make_example(
    sentence: str,
    entity1_span: tuple[int, int],
    entity2_span: tuple[int, int],
) -> SemEvalExample:
    return SemEvalExample(
        index_of_sent=0,
        sentence=sentence,
        entity1_span=entity1_span,
        entity2_span=entity2_span,
        entity1=sentence[entity1_span[0]:entity1_span[1]],
        entity2=sentence[entity2_span[0]:entity2_span[1]],
        relation="Other",
    )


def spans_text(sentence: str, span: tuple[int, int]) -> str:
    start, end = span
    return sentence[start:end]


@pytest.fixture
def simple_example() -> SemEvalExample:
    return make_example(
        sentence="The virus causes the disease in humans.",
        entity1_span=(4, 9),
        entity2_span=(21, 28),
    )


@pytest.fixture
def reversed_example() -> SemEvalExample:
    return make_example(
        sentence="The disease is caused by the virus.",
        entity1_span=(29, 34),
        entity2_span=(4, 11),
    )


@pytest.fixture
def example_with_comma_after_e1() -> SemEvalExample:
    return make_example(
        sentence="The viruses, only four in number, were found.",
        entity1_span=(4, 11),
        entity2_span=(18, 22),
    )


@pytest.fixture
def example_with_possessive() -> SemEvalExample:
    return make_example(
        sentence="The foundry's major lines were sold.",
        entity1_span=(4, 11),
        entity2_span=(20, 25),
    )


@pytest.fixture
def example_with_ascii_double_quote() -> SemEvalExample:
    return make_example(
        sentence='The "virus" causes the disease in humans.',
        entity1_span=(5, 10),
        entity2_span=(23, 30),
    )


@pytest.fixture
def example_with_typographic_quote() -> SemEvalExample:
    return make_example(
        sentence="The \u201cvirus\u201d causes the disease in humans.",
        entity1_span=(5, 10),
        entity2_span=(23, 30),
    )


class TestEntityTextPreserved:
    @pytest.mark.parametrize("template", NOISE_TEMPLATES)
    def test_e1_text_preserved(self, simple_example, template):
        noisy, skipped = insert_noise(simple_example, template)

        assert not skipped
        assert spans_text(noisy.sentence, noisy.entity1_span) == "virus"

    @pytest.mark.parametrize("template", NOISE_TEMPLATES)
    def test_e2_text_preserved(self, simple_example, template):
        noisy, skipped = insert_noise(simple_example, template)

        assert not skipped
        assert spans_text(noisy.sentence, noisy.entity2_span) == "disease"

    @pytest.mark.parametrize("template", NOISE_TEMPLATES)
    def test_both_entity_texts_preserved_when_e2_before_e1(
        self, reversed_example, template
    ):
        noisy, skipped = insert_noise(reversed_example, template)

        assert not skipped
        assert spans_text(noisy.sentence, noisy.entity1_span) == "virus"
        assert spans_text(noisy.sentence, noisy.entity2_span) == "disease"


class TestCommaHandling:
    @pytest.mark.parametrize("template", NOISE_TEMPLATES)
    def test_no_double_comma_when_entity_followed_by_comma(
        self, example_with_comma_after_e1, template
    ):
        noisy, skipped = insert_noise(example_with_comma_after_e1, template)

        assert not skipped
        assert ",," not in noisy.sentence

    @pytest.mark.parametrize("template", NOISE_TEMPLATES)
    def test_trailing_comma_trimmed_when_entity_followed_by_comma(
        self, example_with_comma_after_e1, template
    ):
        trimmed = template[:-1]

        noisy, skipped = insert_noise(example_with_comma_after_e1, template)

        assert not skipped
        e1_end = example_with_comma_after_e1.entity1_span[1]
        assert noisy.sentence[e1_end:e1_end + len(trimmed)] == trimmed
        assert noisy.sentence[e1_end + len(trimmed)] == ","

    @pytest.mark.parametrize("template", NOISE_TEMPLATES)
    def test_full_template_kept_when_no_comma_after_entity(
        self, simple_example, template
    ):
        noisy, skipped = insert_noise(simple_example, template)

        assert not skipped
        e1_end = simple_example.entity1_span[1]
        assert noisy.sentence[e1_end:e1_end + len(template)] == template


class TestSkipConditions:
    @pytest.mark.parametrize("template", NOISE_TEMPLATES)
    def test_skipped_when_possessive_follows_entity(
        self, example_with_possessive, template
    ):
        _, skipped = insert_noise(example_with_possessive, template)

        assert skipped

    @pytest.mark.parametrize("template", NOISE_TEMPLATES)
    def test_returns_unchanged_example_when_possessive(
        self, example_with_possessive, template
    ):
        noisy, skipped = insert_noise(example_with_possessive, template)

        assert skipped
        assert noisy == example_with_possessive

    @pytest.mark.parametrize("template", NOISE_TEMPLATES)
    def test_skipped_when_ascii_double_quote_follows_entity(
        self, example_with_ascii_double_quote, template
    ):
        _, skipped = insert_noise(example_with_ascii_double_quote, template)

        assert skipped

    @pytest.mark.parametrize("template", NOISE_TEMPLATES)
    def test_skipped_when_typographic_quote_follows_entity(
        self, example_with_typographic_quote, template
    ):
        _, skipped = insert_noise(example_with_typographic_quote, template)

        assert skipped


class TestSpanShifting:
    def test_e1_span_unchanged_when_e1_is_earlier(self, simple_example):
        noisy, skipped = insert_noise(simple_example, NOISE_TEMPLATES[0])

        assert not skipped
        assert noisy.entity1_span == simple_example.entity1_span

    def test_e2_span_shifted_by_template_length_when_e1_is_earlier(
        self, simple_example
    ):
        template = NOISE_TEMPLATES[0]
        delta = len(template)

        noisy, skipped = insert_noise(simple_example, template)

        assert not skipped
        assert noisy.entity2_span == (
            simple_example.entity2_span[0] + delta,
            simple_example.entity2_span[1] + delta,
        )

    def test_e2_span_unchanged_when_e2_is_earlier(self, reversed_example):
        noisy, skipped = insert_noise(reversed_example, NOISE_TEMPLATES[0])

        assert not skipped
        assert noisy.entity2_span == reversed_example.entity2_span

    def test_e1_span_shifted_by_template_length_when_e2_is_earlier(
        self, reversed_example
    ):
        template = NOISE_TEMPLATES[0]
        delta = len(template)

        noisy, skipped = insert_noise(reversed_example, template)

        assert not skipped
        assert noisy.entity1_span == (
            reversed_example.entity1_span[0] + delta,
            reversed_example.entity1_span[1] + delta,
        )

    def test_shift_uses_trimmed_template_length_when_comma_follows(
        self, example_with_comma_after_e1
    ):
        template = NOISE_TEMPLATES[0]
        trimmed_delta = len(template) - 1

        noisy, skipped = insert_noise(example_with_comma_after_e1, template)

        assert not skipped
        assert noisy.entity2_span == (
            example_with_comma_after_e1.entity2_span[0] + trimmed_delta,
            example_with_comma_after_e1.entity2_span[1] + trimmed_delta,
        )

    def test_sentence_length_increases_by_template_length(self, simple_example):
        template = NOISE_TEMPLATES[0]

        noisy, _ = insert_noise(simple_example, template)

        assert len(noisy.sentence) == len(simple_example.sentence) + len(template)

    def test_sentence_length_unchanged_on_skip(self, example_with_possessive):
        noisy, skipped = insert_noise(
            example_with_possessive, NOISE_TEMPLATES[0]
        )

        assert skipped
        assert len(noisy.sentence) == len(example_with_possessive.sentence)


class TestMetadataPreserved:
    def test_other_fields_are_preserved_via_replace(self, simple_example):
        example = replace(
            simple_example,
            index_of_sent=42,
            relation="Cause-Effect(e1,e2)",
        )

        noisy, skipped = insert_noise(example, NOISE_TEMPLATES[0])

        assert not skipped
        assert noisy.index_of_sent == 42
        assert noisy.relation == "Cause-Effect(e1,e2)"

    def test_other_fields_are_preserved_on_skip(self, example_with_possessive):
        example = replace(
            example_with_possessive,
            index_of_sent=7,
            relation="Product-Producer(e1,e2)",
        )

        noisy, skipped = insert_noise(example, NOISE_TEMPLATES[0])

        assert skipped
        assert noisy.index_of_sent == 7
        assert noisy.relation == "Product-Producer(e1,e2)"