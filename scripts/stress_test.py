import argparse
import random
from dataclasses import replace
from pathlib import Path

from scripts.prepare_data import write_jsonl
from src.relation_extraction.config import SEED
from src.relation_extraction.data.semeval_loader import SemEvalLoader
from src.relation_extraction.models.stress_stats import StressStats
from src.relation_extraction.parsing.parser import Parser
from src.relation_extraction.pipeline.generate_sample import build_record


ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT / "data" / "raw" / "TEST_FILE_FULL.TXT"
OUTPUT_PATH = ROOT / "data" / "processed" / "test_noisy.jsonl"
RESULTS_DIR = ROOT / "results"

NOISE_TEMPLATES = [
    ", according to the report,",
    ", which was mentioned earlier,",
    ", as reported in the study,",
    ", according to the available information,",
]

QUOTE_CHARS = frozenset({
    '"', "'",
    "\u2018", "\u2019",
    "\u201c", "\u201d",
    "\u00ab", "\u00bb",
    "\u201e",
})


def insert_noise(example, template: str):
    e1_start, e1_end = example.entity1_span
    e2_start, e2_end = example.entity2_span

    if e1_start < e2_start:
        earlier_end = e1_end
        shift_entity2 = True
    else:
        earlier_end = e2_end
        shift_entity2 = False

    next_char = example.sentence[earlier_end:earlier_end + 1]

    if next_char in QUOTE_CHARS:
        return example, True

    if next_char == "," and template.endswith(","):
        template = template[:-1]

    delta = len(template)

    if shift_entity2:
        new_e1_span = example.entity1_span
        new_e2_span = (e2_start + delta, e2_end + delta)
    else:
        new_e1_span = (e1_start + delta, e1_end + delta)
        new_e2_span = example.entity2_span

    noisy_sentence = (
        example.sentence[:earlier_end]
        + template
        + example.sentence[earlier_end:]
    )

    noisy_example = replace(
        example,
        sentence=noisy_sentence,
        entity1_span=new_e1_span,
        entity2_span=new_e2_span,
    )
    return noisy_example, False


def word_signature(record):
    path = record["sdp"]
    if path is None:
        return None
    return tuple(record["tokens"][i].casefold() for i in path)


def get_path_signature(record: dict):
    words = word_signature(record)
    if words is None:
        return None
    return tuple(zip(words, [None] + record["sdp_deps"]))


def path_hits_insertion(record, offsets, start, end):
    """Есть ли на пути слово из символьного интервала [start, end)?"""
    return any(
        start <= offsets[i] < end
        and any(char.isalnum() for char in record["tokens"][i])
        for i in record["sdp"] or ()
    )


def format_path(record):
    words = [record["tokens"][i] for i in record["sdp"]]
    parts = [words[0]]

    for dependency, word in zip(record["sdp_deps"], words[1:], strict=True):
        parts.extend([f"[{dependency}]", word])

    return " ".join(parts)


def ratio(count, total):
    if total == 0:
        return "0/0 (н/д)"
    return f"{count}/{total} ({100 * count / total:.2f}%)"


def run_stress_test(examples, parser, loader, rng) -> tuple[list[dict], StressStats]:
    noisy_records = []
    stats = StressStats()

    for example in examples:
        original_record, original_ok = build_record(example, parser, loader)

        if not original_ok:
            stats.original_failed += 1
        elif original_record["sdp"] is None:
            stats.original_no_path += 1

        template = rng.choice(NOISE_TEMPLATES)
        stats.template_usage[template] += 1

        noisy_example, skipped = insert_noise(example, template)
        noisy_record, noisy_ok = build_record(
            noisy_example, parser, loader, include_offsets=True
        )
        offsets = noisy_record.pop("offsets")
        noisy_records.append(noisy_record)

        if skipped:
            stats.insertion_skipped += 1
            continue

        if not noisy_ok:
            stats.noisy_failed += 1
        elif noisy_record["sdp"] is None:
            stats.noisy_no_path += 1

        if not original_ok or not noisy_ok:
            continue
        if original_record["sdp"] is None or noisy_record["sdp"] is None:
            continue

        stats.comparable += 1
        stats.template_comparable[template] += 1

        if word_signature(original_record) == word_signature(noisy_record):
            stats.unchanged_words += 1

        if get_path_signature(original_record) == get_path_signature(noisy_record):
            stats.unchanged += 1
            stats.template_unchanged[template] += 1
            continue

        start = min(example.entity1_span[1], example.entity2_span[1])
        end = start + len(noisy_example.sentence) - len(example.sentence)
        hit = path_hits_insertion(noisy_record, offsets, start, end)
        stats.changed_through_insert += int(hit)

        if len(stats.examples) < 15:
            stats.examples.append(
                f"## ID {example.index_of_sent}\n\n"
                f"{example.sentence}\n\n"
                f"После вставки — {noisy_example.sentence}\n\n"
                f"До: `{format_path(original_record)}`\n\n"
                f"После: `{format_path(noisy_record)}`\n\n"
                f"Через вставленное слово — {'да' if hit else 'нет'}."
            )

    return noisy_records, stats


def print_report(stats: StressStats, total: int) -> None:
    changed = stats.comparable - stats.unchanged

    print("\nСтресс-тест завершён.")
    print(f"Всего предложений: {total}")
    print(f"Вставка пропущена (пунктуация после сущности): {stats.insertion_skipped}")

    print("\nСтатистика построения SDP:")
    print(f"  Нет SDP в исходных предложениях: {stats.original_no_path}")
    print(f"  Нет SDP в изменённых предложениях: {stats.noisy_no_path}")
    print(f"  Ошибки определения сущностей (исходные): {stats.original_failed}")
    print(f"  Ошибки определения сущностей (изменённые): {stats.noisy_failed}")

    print("\nСравнение кратчайших путей (пропущенные вставки не учитываются):")
    print(f"  Предложений, пригодных для сравнения: {stats.comparable}")
    print(f"  SDP не изменился: {stats.unchanged}")
    print(f"  SDP изменился: {changed}")
    print(f"  Устойчивость SDP (слова и связи): {ratio(stats.unchanged, stats.comparable)}")
    print(f"  Устойчивость по словам: {ratio(stats.unchanged_words, stats.comparable)}")
    print(f"  Через вставленные слова среди изменившихся: {ratio(stats.changed_through_insert, changed)}")

    print("\nУстойчивость по шаблонам (слова и связи):")
    for template in NOISE_TEMPLATES:
        score = ratio(stats.template_unchanged[template], stats.template_comparable[template])
        print(f"  {template}: выбрано {stats.template_usage[template]}, устойчивость {score}")


def main() -> None:
    cli = argparse.ArgumentParser(description="Сравнить SDP до и после вставки")
    cli.add_argument(
        "--spacy-model",
        choices=["en_core_web_sm", "en_core_web_lg"],
        default="en_core_web_sm",
    )
    args = cli.parse_args()

    loader = SemEvalLoader()
    parser = Parser(args.spacy_model)
    rng = random.Random(SEED)

    examples = loader.load_file(str(TEST_PATH))
    noisy_records, stats = run_stress_test(examples, parser, loader, rng)

    if args.spacy_model == "en_core_web_sm":
        write_jsonl(OUTPUT_PATH, noisy_records)
        print(f"Сохранено предложений: {len(noisy_records)}")
        print(f"Файл: {OUTPUT_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_size = args.spacy_model.rsplit("_", 1)[-1]
    examples_path = RESULTS_DIR / f"stress_examples({model_size}).md"
    header = f"# Изменения SDP · {args.spacy_model}"
    examples_path.write_text(
        "\n\n".join([header, *stats.examples]) + "\n",
        encoding="utf-8",
    )

    print(f"Модель: {args.spacy_model}. Примеры: {examples_path}")
    print_report(stats, total=len(examples))


if __name__ == "__main__":
    main()