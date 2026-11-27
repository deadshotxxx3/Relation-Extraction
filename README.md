Relation-Extraction

## Нейросетевые модели

Запуск из корня репозитория после `python -m scripts.prepare_data` и `python -m scripts.stress_test`.
GloVe: из архива `glove.6B.zip` (https://nlp.stanford.edu/projects/glove/) положить `glove.6B.100d.txt` в `data/glove/`.

```bash
# обучение: BiLSTM по предложению и по пути, GCN по дереву
python -m scripts.train_nn --model bilstm --input sentence --train data/processed/train.jsonl --dev data/processed/dev.jsonl --test data/processed/test.jsonl --glove data/glove/glove.6B.100d.txt
python -m scripts.train_nn --model bilstm --input sdp      --train data/processed/train.jsonl --dev data/processed/dev.jsonl --test data/processed/test.jsonl --glove data/glove/glove.6B.100d.txt
python -m scripts.train_nn --model gcn    --input sentence --train data/processed/train.jsonl --dev data/processed/dev.jsonl --test data/processed/test.jsonl --glove data/glove/glove.6B.100d.txt

# готовая модель на зашумлённом тесте, без переобучения
python -m scripts.train_nn --checkpoint models/bilstm_sdp.pt --test data/processed/test_noisy.jsonl
```

Модели сохраняются в `models/<model>_<input>.pt`, метрики пишет `evaluate` в `results/`.
