from src.relation_extraction.baseline import (
    load_split,
    train_baseline,
    predict_baseline,
)

from src.relation_extraction.evaluate import evaluate


MODES = ["sentence", "sdp", "sdp_deps"]


def run_baselines():
    for mode in MODES:
        print(f"\nRunning baseline: {mode}")

        train_ids, train_texts, train_labels = load_split(
            "data/processed/train.jsonl",
            mode,
        )

        dev_ids, dev_texts, dev_labels = load_split(
            "data/processed/dev.jsonl",
            mode,
        )

        test_ids, test_texts, test_labels = load_split(
            "data/processed/test.jsonl",
            mode,
        )

        model = train_baseline(
            train_texts,
            train_labels,
        )

        dev_predictions = predict_baseline(model, dev_texts)
        test_predictions = predict_baseline(model, test_texts)

        evaluate(
            ids=dev_ids,
            gold=dev_labels,
            pred=dev_predictions,
            name=f"baseline_{mode}",
            split="dev",
        )

        evaluate(
            ids=test_ids,
            gold=test_labels,
            pred=test_predictions,
            name=f"baseline_{mode}",
            split="test",
        )


if __name__ == "__main__":
    run_baselines()