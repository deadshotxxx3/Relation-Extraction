import copy

import torch
import torch.nn as nn

from src.relation_extraction.config import LABELS
from src.relation_extraction.evaluate import official_f1


def get_device():
    """GPU, если он есть (например на Kaggle), иначе процессор."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_batch(batch, device):
    """Разбирает пачку из collate на входы модели и метки."""
    ids = batch[0].to(device)
    lengths = batch[1]
    e1 = batch[2].to(device)
    e2 = batch[3].to(device)
    labels = batch[4].to(device)
    return ids, lengths, e1, e2, labels


def run_epoch(model, loader, loss_fn, optimizer=None, device="cpu", clip=5.0):
    """Один проход по loader. С optimizer — обучение, без него - только средняя ошибка."""
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    count = 0
    for batch in loader:
        ids, lengths, e1, e2, labels = split_batch(batch, device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(ids, lengths, e1, e2)
            loss = loss_fn(logits, labels)
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        total_loss += loss.item() * len(labels)
        count += len(labels)
    return total_loss / count


def predict(model, loader, device="cpu"):
    """Предсказанные метки (строки из LABELS) в порядке примеров loader."""
    model.eval()
    result = []
    with torch.no_grad():
        for batch in loader:
            ids, lengths, e1, e2, labels = split_batch(batch, device)
            logits = model(ids, lengths, e1, e2)
            for label_id in logits.argmax(dim=-1).tolist():
                result.append(LABELS[label_id])
    return result


def accuracy(gold, pred):
    """Доля совпавших меток."""
    if len(gold) == 0:
        return 0.0
    correct = 0
    for g, p in zip(gold, pred):
        if g == p:
            correct += 1
    return correct / len(gold)


def train(model, train_loader, dev_loader, epochs=30, lr=1e-3, weight_decay=0.01,
          patience=5, device="cpu"):
    """Обучает модель и оставляет в ней веса лучшей по dev эпохи.

    Лучшая эпоха выбирается по официальному macro-F1 SemEval на dev.
    Если F1 не растёт patience эпох подряд, обучение останавливается
    (patience = 0 — учить все epochs эпох).
    dev_loader должен идти без перемешивания (shuffle=False), его dataset - REDataset.
    Возвращает историю: список словарей по эпохам."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    dev_gold = dev_loader.dataset.gold_labels()

    best_f1 = -1.0
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, device)
        dev_pred = predict(model, dev_loader, device)
        dev_f1, per_relation = official_f1(dev_gold, dev_pred)
        dev_acc = accuracy(dev_gold, dev_pred)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_f1": dev_f1,
            "dev_acc": dev_acc,
        })
        print(
            "epoch " + str(epoch)
            + "  loss " + format(train_loss, ".4f")
            + "  dev F1 " + format(dev_f1, ".2f")
            + "  dev acc " + format(dev_acc * 100, ".2f")
        )

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience > 0 and bad_epochs >= patience:
                print("no improvement for " + str(patience) + " epochs, stopping")
                break

    model.load_state_dict(best_state)
    print("best dev F1 " + format(best_f1, ".2f"))
    return history
