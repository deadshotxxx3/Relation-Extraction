"""Точка входа: генерирует tests/sample.jsonl из SemEval TRAIN_FILE.TXT."""
from src.relation_extraction.pipeline.generate_sample import generate_sample


INPUT_PATH = "./data/raw/TRAIN_FILE.TXT"
OUTPUT_PATH = "tests/sample.jsonl"
LIMIT = 20


def main() -> None:
    stats = generate_sample(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        limit=LIMIT,
    )

    print(f"Wrote {stats['written']} records -> {OUTPUT_PATH}")

    if stats["failed_entity_ids"]:
        print(
            f"Entity resolution failed for {len(stats['failed_entity_ids'])} "
            f"examples: {stats['failed_entity_ids']}"
        )

    print(f"Examples without dependency path: {stats['no_path']}")


if __name__ == "__main__":
    main()