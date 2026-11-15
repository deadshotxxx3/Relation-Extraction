import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.relation_extraction.config import LABELS, SEED
from src.relation_extraction.evaluate import evaluate
from src.relation_extraction.nn.bilstm import BiLSTMClassifier
from src.relation_extraction.nn.data_utils import build_vocab, load_jsonl
from src.relation_extraction.nn.dataset import REDataset, collate
from src.relation_extraction.nn.train import get_device, predict, train


def parse_args():
    parser = argparse.ArgumentParser(description="Train BiLSTM relation classifier")
    parser.add_argument("--model", choices=["bilstm"], default="bilstm")
    parser.add_argument("--input", choices=["sentence", "sdp"], default="sentence")
    parser.add_argument("--train", help="jsonl для обучения")
    parser.add_argument("--dev", help="jsonl для выбора лучшей эпохи; если нет — берётся train")
    parser.add_argument("--test", nargs="*", default=[], help="один или несколько jsonl для оценки")
    parser.add_argument("--name", help="имя модели в файлах results/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--emb-dim", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5, help="0 — без ранней остановки")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def make_model(settings, vocab_size):
    return BiLSTMClassifier(
        vocab_size,
        len(LABELS),
        emb_dim=settings["emb_dim"],
        hidden=settings["hidden"],
        dropout=settings["dropout"],
    )


def make_loader(dataset, settings, shuffle):
    return DataLoader(dataset, batch_size=settings["batch_size"], shuffle=shuffle,
                      collate_fn=collate)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    if args.train is None:
        raise SystemExit("--train is required")

    settings = {
        "model": args.model,
        "input": args.input,
        "emb_dim": args.emb_dim,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
    }
    name = args.name
    if name is None:
        name = args.model + "_" + args.input

    vocab = build_vocab(load_jsonl(args.train))
    print("vocab size " + str(len(vocab)))

    dev_path = args.dev
    if dev_path is None:
        print("no --dev given, using train as dev (only for the overfit check)")
        dev_path = args.train

    train_set = REDataset(args.train, vocab, args.input)
    dev_set = REDataset(dev_path, vocab, args.input)
    print("train " + str(len(train_set)) + " examples, dev " + str(len(dev_set)))

    model = make_model(settings, len(vocab))
    train_loader = make_loader(train_set, settings, True)
    dev_loader = make_loader(dev_set, settings, False)

    train(model, train_loader, dev_loader, epochs=args.epochs, lr=args.lr,
          weight_decay=args.weight_decay, patience=args.patience, device=device)

    for test_path in args.test:
        test_set = REDataset(test_path, vocab, settings["input"])
        loader = make_loader(test_set, settings, False)
        pred = predict(model, loader, device)
        split = Path(test_path).stem
        evaluate(test_set.ids(), test_set.gold_labels(), pred, name, split)


if __name__ == "__main__":
    main()
