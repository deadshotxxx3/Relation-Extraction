import random
from dataclasses import replace
from pathlib import Path

from src.relation_extraction.config import SEED
from src.relation_extraction.data.semeval_loader import SemEvalLoader
from src.relation_extraction.parsing.parser import Parser
from src.relation_extraction.pipeline.generate_sample import build_record
from src.relation_extraction.models.stress_stats import StressStats

from scripts.prepare_data import write_jsonl


ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT / "data" / "raw" / "TEST_FILE_FULL.TXT"
OUTPUT_PATH = ROOT / "data" / "processed" / "test_noisy.jsonl"

NOISE_TEMPLATES = [
    ", according to the report,",
    ", which was mentioned earlier,",
    ", as reported in the study,",
    ", according to the available information,",
]

QUOTE_CHARS = frozenset({
    '"', "'",                  # ASCII
    "\u2018", "\u2019",        # ' '
    "\u201c", "\u201d",        # " "
    "\u00ab", "\u00bb",        # « »
    "\u201e",                  # „
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

    noisy_sentence = (example.sentence[:earlier_end]
        + template + example.sentence[earlier_end:])

    noisy_example = replace(example, sentence=noisy_sentence,
        entity1_span=new_e1_span, entity2_span=new_e2_span)

    return noisy_example, False


def get_path_signature(record: dict):
    if record["sdp"] is None:
        return None

    tokens = record["tokens"]
    path = record["sdp"]
    dependencies = record["sdp_deps"]

    path_words = [tokens[index].casefold() for index in path]

    return tuple(zip(path_words, [None] + dependencies))


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
        if skipped:
            stats.insertion_skipped += 1

        noisy_record, noisy_ok = build_record(noisy_example, parser, loader)
        noisy_records.append(noisy_record)

        if skipped:
            continue

        if not noisy_ok:
            stats.noisy_failed += 1
        elif noisy_record["sdp"] is None:
            stats.noisy_no_path += 1

        if (original_ok and noisy_ok
                and original_record["sdp"] is not None
                and noisy_record["sdp"] is not None):
            stats.comparable += 1

            original_signature = get_path_signature(original_record)
            noisy_signature = get_path_signature(noisy_record)

            if original_signature == noisy_signature:
                stats.unchanged += 1

    return noisy_records, stats


def print_report(stats: StressStats, total: int, output_path: Path) -> None:
    changed = stats.comparable - stats.unchanged
    robustness = (
        stats.unchanged / stats.comparable * 100
        if stats.comparable > 0 else 0
    )

    print("\nСтресс-тест завершён.")
    print(f"Всего предложений: {total}")
    print(f"Вставка пропущена (пунктуация после сущности): {stats.insertion_skipped}")
    print(f"Сохранено предложений: {total}")
    print(f"Файл: {output_path}")

    print("\nСтатистика построения SDP:")
    print(f"  Нет SDP в исходных предложениях: {stats.original_no_path}")
    print(f"  Нет SDP в изменённых предложениях: {stats.noisy_no_path}")
    print(f"  Ошибки определения сущностей (исходные): {stats.original_failed}")
    print(f"  Ошибки определения сущностей (изменённые): {stats.noisy_failed}")

    print("\nСравнение кратчайших путей (пропущенные вставки не учитываются):")
    print(f"  Предложений, пригодных для сравнения: {stats.comparable}")
    print(f"  SDP не изменился: {stats.unchanged}")
    print(f"  SDP изменился: {changed}")
    print(f"  Устойчивость SDP: {robustness:.2f}%")

    print("\nИспользование шаблонов:")
    for template, count in stats.template_usage.items():
        print(f"  {template}: {count}")


def main() -> None:
    loader = SemEvalLoader()
    parser = Parser()
    rng = random.Random(SEED)

    examples = loader.load_file(str(TEST_PATH))

    noisy_records, stats = run_stress_test(examples, parser, loader, rng)

    write_jsonl(OUTPUT_PATH, noisy_records)
    print_report(stats, total=len(examples), output_path=OUTPUT_PATH)


if __name__ == "__main__":
    main()
