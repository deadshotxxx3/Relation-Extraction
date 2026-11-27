import argparse
from pathlib import Path
from src.relation_extraction.evaluate import evaluate, official_f1

from src.relation_extraction.baseline import (
    load_split,
    train_baseline,
    predict_baseline,
)


MODES = ["sentence", "sdp", "sdp_deps"]
C_VALUES = [0.1, 1, 10]


def run_baselines(mode, train_path, dev_path, test_paths, C):
    print(f"\nRunning baseline: {mode}")

    train_ids, train_texts, train_labels = load_split(
        train_path, mode
    )

    dev_ids, dev_texts, dev_labels = load_split(
        dev_path, mode
    )

    best_C = None
    best_score = -1.0
    best_model = None
    
    c_values = C if C is not None else C_VALUES

    for current_C in c_values:
        model = train_baseline(
            train_texts,
            train_labels,
            C=current_C,
        )

        dev_predictions = predict_baseline(model, dev_texts)

        score, _ = official_f1(dev_labels, dev_predictions)

        print(f"C={current_C}: dev F1 = {score:.2f}")

        if score > best_score:
            best_score = score
            best_C = current_C
            best_model = model

    print(f"Selected C = {best_C}")

    for test_path in test_paths:
        test_ids, test_texts, test_labels = load_split(
            test_path, mode
        )

        predictions = predict_baseline(best_model, test_texts)

        evaluate(
            ids=test_ids,
            gold=test_labels,
            pred=predictions,
            name=f"baseline_{mode}",
            split=Path(test_path).stem,
        )


def main(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test", nargs="+", required=True)
    parser.add_argument("--C", nargs="+", type=float, default=None)

    args = parser.parse_args(argv)

    run_baselines(
        mode=args.mode,
        train_path=args.train,
        dev_path=args.dev,
        test_paths=args.test,
        C=args.C,
    )


if __name__ == "__main__":
    main()
