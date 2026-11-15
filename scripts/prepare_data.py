
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.relation_extraction.config import LABELS, LABEL_TO_ID, RANDOM_STATE
from src.relation_extraction.data.semeval_loader import SemEvalLoader
from src.relation_extraction.parsing.parser import Parser
from src.relation_extraction.pipeline.generate_sample import build_record


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "raw" / "TRAIN_FILE.TXT"
TEST_PATH = ROOT / "data" / "raw" / "TEST_FILE_FULL.TXT"
OUTPUT_DIR = ROOT / "data" / "processed"

DEV_SIZE = 800

def base_relation(label: str) -> str:
    return label.split("(", maxsplit=1)[0]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_split(examples, parser, loader):
    records = []
    no_path = 0
    failed_entity_ids = []

    for example in examples:
        record, entities_ok = build_record(example, parser, loader)

        records.append(record)

        if not entities_ok:
            failed_entity_ids.append(example.index_of_sent)
        elif record["sdp"] is None:
            no_path += 1

    return records, no_path, failed_entity_ids


def validate_records(name: str, records: list[dict]) -> None:
    for record in records:
        size = len(record["tokens"])

        for field in ("heads", "deprels", "pos"):
            if len(record[field]) != size:
                raise ValueError(
                    f"{name}: example {record['id']}: "
                    f"len(tokens) != len({field})"
                )

        label = record["label"]

        if label not in LABEL_TO_ID:
            raise ValueError(
                f"{name}: unknown label {label!r} "
                f"for example {record['id']}"
            )

        if record["label_id"] != LABEL_TO_ID[label]:
            raise ValueError(
                f"{name}: incorrect label_id for example {record['id']}"
            )

        if record["label_id"] not in range(len(LABELS)):
            raise ValueError(
                f"{name}: invalid label_id {record['label_id']} "
                f"for example {record['id']}"
            )


def print_statistics(
    name: str,
    records: list[dict],
    no_path: int,
    failed_entity_ids: list[int],
) -> None:
    sentence_lengths = [
        len(record["tokens"])
        for record in records
    ]

    path_lengths = [
        len(record["sdp"])
        for record in records
        if record["sdp"] is not None
    ]

    distribution = Counter(path_lengths)

    average_sentence = 0
    average_path = 0

    if sentence_lengths:
        average_sentence = sum(sentence_lengths) / len(sentence_lengths)

    if path_lengths:
        average_path = sum(path_lengths) / len(path_lengths)

    print(f"\n{name}:")
    print(f"  Примеров: {len(records)}")
    print(f"  Без shortest dependency path: {no_path}")
    print(f"  Не удалось найти токены сущностей: {len(failed_entity_ids)}")
    print(f"  Средняя длина предложения (токены): {average_sentence:.2f}")
    print(f"  Средняя длина пути (вершины): {average_path:.2f}")

    print("  Распределение длины пути (вершины):")

    for length in sorted(distribution):
        print(f"    {length}: {distribution[length]}")


def main() -> None:
    loader = SemEvalLoader()
    parser = Parser()

    train_examples = loader.load_file(str(TRAIN_PATH))
    test_examples = loader.load_file(str(TEST_PATH))

    if len(train_examples) <= DEV_SIZE:
        raise ValueError(
            "Количество обучающих примеров должно быть больше размера DEV"
        )

    stratify_labels = [
        base_relation(example.relation)
        for example in train_examples
    ]

    train_examples, dev_examples = train_test_split(
        train_examples,
        test_size=DEV_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
    )

    splits = {
        "train": train_examples,
        "dev": dev_examples,
        "test": test_examples,
    }

    print(f"Количество меток: {len(LABELS)}")
    print(f"Random state: {RANDOM_STATE}")

    print(
        f"Разбиение: train={len(train_examples)}, "
        f"dev={len(dev_examples)}, "
        f"train + dev={len(train_examples) + len(dev_examples)}, "
        f"test={len(test_examples)}"
    )

    for split_name, examples in splits.items():
        records, no_path, failed_entity_ids = prepare_split(
            examples,
            parser,
            loader,
        )

        validate_records(split_name, records)

        output_path = OUTPUT_DIR / f"{split_name}.jsonl"
        write_jsonl(output_path, records)

        print(f"\nСохранено: {output_path}")

        print_statistics(
            split_name,
            records,
            no_path,
            failed_entity_ids,
        )

    print("\nВсе проверки успешно пройдены.")
    print("  Проверены label_id и соответствие меток.")
    print("  Проверены длины tokens, heads, deprels и pos.")
    print(
        f"  Общее количество train и dev: "
        f"{len(train_examples) + len(dev_examples)}"
    )
    print(f"  Количество test: {len(test_examples)}")


if __name__ == "__main__":
    main()
