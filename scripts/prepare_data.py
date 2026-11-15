import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

from src.relation_extraction.config import RANDOM_STATE
from src.relation_extraction.data.semeval_loader import SemEvalLoader
from src.relation_extraction.parsing.parser import Parser
from src.relation_extraction.pipeline.generate_sample import build_record


ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT / "data" / "raw" / "TEST_FILE_FULL.TXT"
OUTPUT_PATH = ROOT / "data" / "processed" / "test_noisy.jsonl"

NOISE_TEMPLATES = [
    ", according to the report,",
    ", which was mentioned earlier,",
    ", as reported in the study,",
    ", according to the available information,",
]


def insert_noise(example, template: str):
    e1_start, e1_end = example.entity1_span
    e2_start, e2_end = example.entity2_span

    if e1_start < e2_start:
        earlier_end = e1_end

        new_e1_span = example.entity1_span
        new_e2_span = (e2_start + len(template), e2_end + len(template))
    else:
        earlier_end = e2_end

        new_e1_span = (e1_start + len(template), e1_end + len(template))
        new_e2_span = example.entity2_span

    noisy_sentence = (
        example.sentence[:earlier_end] + template + example.sentence[earlier_end:])

    noisy_example = replace(example, sentence=noisy_sentence,
        entity1_span=new_e1_span, entity2_span=new_e2_span)

    return noisy_example


def get_path_signature(record: dict):
    if record["sdp"] is None:
        return None

    tokens = record["tokens"]
    path = record["sdp"]
    dependencies = record["sdp_deps"]

    path_words = [tokens[index].casefold() for index in path]

    return tuple(zip(path_words, [None] + dependencies))


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def main():
    loader = SemEvalLoader()
    parser = Parser()
    rng = random.Random(RANDOM_STATE)

    examples = loader.load_file(str(TEST_PATH))

    noisy_records = []

    original_no_path = 0
    noisy_no_path = 0

    original_failed = 0
    noisy_failed = 0

    comparable = 0
    unchanged = 0

    template_usage = Counter()

    for example in examples:
        original_record, original_ok = build_record(example, parser, loader)

        if not original_ok:
            original_failed += 1
        elif original_record["sdp"] is None:
            original_no_path += 1

        template = rng.choice(NOISE_TEMPLATES)
        template_usage[template] += 1

        noisy_example = insert_noise(example, template)

        noisy_record, noisy_ok = build_record(noisy_example, parser, loader)

        noisy_records.append(noisy_record)

        if not noisy_ok:
            noisy_failed += 1
        elif noisy_record["sdp"] is None:
            noisy_no_path += 1

        if (original_ok and noisy_ok
            and original_record["sdp"] is not None
            and noisy_record["sdp"] is not None):
            comparable += 1

            original_signature = get_path_signature(original_record)
            noisy_signature = get_path_signature(noisy_record)

            if original_signature == noisy_signature:
                unchanged += 1

    write_jsonl(OUTPUT_PATH, noisy_records)

    changed = comparable - unchanged

    robustness = (unchanged / comparable * 100 if comparable > 0 else 0)

    print("\nСтресс-тест завершён.")
    print(f"Всего предложений: {len(examples)}")
    print(f"Сохранено предложений: {len(noisy_records)}")
    print(f"Файл: {OUTPUT_PATH}")

    print("\nСтатистика построения SDP:")
    print(f"  Нет SDP в исходных предложениях: {original_no_path}")
    print(f"  Нет SDP в изменённых предложениях: {noisy_no_path}")
    print(f"  Ошибки определения сущностей (исходные): {original_failed}")
    print(f"  Ошибки определения сущностей (изменённые): {noisy_failed}")

    print("\nСравнение кратчайших путей:")
    print(f"  Предложений, пригодных для сравнения: {comparable}")
    print(f"  SDP не изменился: {unchanged}")
    print(f"  SDP изменился: {changed}")
    print(f"  Устойчивость SDP: {robustness:.2f}%")

    print("\nИспользование шаблонов:")
    for template, count in template_usage.items():
        print(f"  {template}: {count}")


if __name__ == "__main__":
    main()
