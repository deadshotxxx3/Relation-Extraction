import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.relation_extraction.config import LABELS, SEED
from src.relation_extraction.evaluate import evaluate
from src.relation_extraction.nn.bilstm import BiLSTMClassifier
from src.relation_extraction.nn.data_utils import build_vocab, load_glove, load_jsonl
from src.relation_extraction.nn.dataset import REDataset, collate, collate_gcn
from src.relation_extraction.nn.gcn import GCNClassifier
from src.relation_extraction.nn.train import get_device, predict, train


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train BiLSTM or GCN relation classifier")
    parser.add_argument("--model", choices=["bilstm", "gcn"], default="bilstm")
    parser.add_argument("--input", choices=["sentence", "sdp"], default="sentence")
    parser.add_argument("--train", help="jsonl для обучения")
    parser.add_argument("--dev", help="jsonl для выбора лучшей эпохи; если нет — берётся train")
    parser.add_argument("--test", nargs="*", default=[], help="один или несколько jsonl для оценки")
    parser.add_argument("--checkpoint", help="не обучать, а загрузить модель из .pt и оценить на --test")
    parser.add_argument("--glove", help="файл GloVe, например data/glove/glove.6B.100d.txt")
    parser.add_argument("--name", help="имя модели в models/ и results/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--emb-dim", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--layers", type=int, default=2, help="слоёв GCN")
    parser.add_argument("--no-lstm", action="store_true", help="GCN без BiLSTM перед ним")
    parser.add_argument("--patience", type=int, default=5, help="0 — без ранней остановки")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def make_model(settings, vocab_size, embeddings=None):
    if settings["model"] == "bilstm":
        return BiLSTMClassifier(
            vocab_size,
            len(LABELS),
            emb_dim=settings["emb_dim"],
            hidden=settings["hidden"],
            dropout=settings["dropout"],
            embeddings=embeddings,
        )
    return GCNClassifier(
        vocab_size,
        len(LABELS),
        emb_dim=settings["emb_dim"],
        hidden=settings["hidden"],
        n_layers=settings["layers"],
        dropout=settings["dropout"],
        embeddings=embeddings,
        use_lstm=settings["use_lstm"],
    )


def make_loader(dataset, settings, shuffle):
    if settings["model"] == "gcn":
        collate_fn = collate_gcn
    else:
        collate_fn = collate
    return DataLoader(dataset, batch_size=settings["batch_size"], shuffle=shuffle,
                      collate_fn=collate_fn)


def train_new_model(args, device):
    if args.train is None:
        raise SystemExit("--train is required when --checkpoint is not given")

    settings = {
        "model": args.model,
        "input": args.input,
        "emb_dim": args.emb_dim,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "layers": args.layers,
        "use_lstm": not args.no_lstm,
        "batch_size": args.batch_size,
    }
    name = args.name
    if name is None:
        name = args.model + "_" + args.input

    vocab = build_vocab(load_jsonl(args.train))
    print("vocab size " + str(len(vocab)))

    embeddings = None
    if args.glove is not None:
        embeddings, found = load_glove(args.glove, vocab, args.emb_dim, args.seed)
        print("GloVe vectors found for " + str(found) + " of " + str(len(vocab)) + " words")

    dev_path = args.dev
    if dev_path is None:
        print("no --dev given, using train as dev (only for the overfit check)")
        dev_path = args.train

    train_set = REDataset(args.train, vocab, args.input)
    dev_set = REDataset(dev_path, vocab, args.input)
    print("train " + str(len(train_set)) + " examples, dev " + str(len(dev_set)))

    model = make_model(settings, len(vocab), embeddings)
    train_loader = make_loader(train_set, settings, True)
    dev_loader = make_loader(dev_set, settings, False)

    train(model, train_loader, dev_loader, epochs=args.epochs, lr=args.lr,
          weight_decay=args.weight_decay, patience=args.patience, device=device)

    Path("models").mkdir(exist_ok=True)
    checkpoint_path = Path("models") / (name + ".pt")
    torch.save({
        "name": name,
        "settings": settings,
        "vocab": vocab,
        "state": model.state_dict(),
    }, checkpoint_path)
    print("saved " + str(checkpoint_path))
    return model, settings, vocab, name


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device)
    settings = checkpoint["settings"]
    vocab = checkpoint["vocab"]
    model = make_model(settings, len(vocab))
    model.load_state_dict(checkpoint["state"])
    model.to(device)
    return model, settings, vocab, checkpoint["name"]


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device()

    if args.checkpoint is not None:
        model, settings, vocab, name = load_model(args.checkpoint, device)
        if args.name is not None:
            name = args.name
    else:
        model, settings, vocab, name = train_new_model(args, device)

    for test_path in args.test:
        test_set = REDataset(test_path, vocab, settings["input"])
        loader = make_loader(test_set, settings, False)
        pred = predict(model, loader, device)
        split = Path(test_path).stem
        evaluate(test_set.ids(), test_set.gold_labels(), pred, name, split)


if __name__ == "__main__":
    main()
