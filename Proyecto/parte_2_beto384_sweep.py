# %% [markdown]
# # BETO 384 - barrido controlado de ajuste final
#
# Este notebook replica la configuracion que mejor funciono hasta ahora:
#
# - `dccuchile/bert-base-spanish-wwm-cased`
# - `SEED = 42`
# - `max_len = 384`
# - `use_fast=False`
# - aumentacion suave del notebook que dio 0.27889
# - sin limpieza adicional del texto
#
# Mejora sobre el notebook anterior: guarda submissions antes del ajuste final y
# despues de 0.25, 0.50, 0.75 y 1.00 epocas sobre 100% de `train.csv`.

# %%
from pathlib import Path
import json
import math
import random
import shutil

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IN_KAGGLE = Path("/kaggle/working").exists()
OUTPUT_DIR = Path("/kaggle/working") if IN_KAGGLE else Path("Proyecto/runs_beto384_sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
RUN_NAME = "beto384_sweep_seed42"
MAX_LEN = 384
AUG_FRACTION = 0.15
VALID_SIZE = 0.12
EPOCHS = 4
LR = 2e-5
BATCH_SIZE = 4
GRAD_ACCUM = 4
FINAL_LR = 8e-6
FINAL_FRACTIONS = [0.25, 0.25, 0.25, 0.25]
SAVE_MODEL_ZIP = False

print("GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), "count:", torch.cuda.device_count())
print("output:", OUTPUT_DIR)

# %%
def find_data_dir():
    candidates = [
        Path("/kaggle/input/datasets/jsmaldo/proyecto-archivos"),
        Path("/kaggle/input/proyecto-archivos"),
        Path.cwd(),
        Path.cwd() / "Proyecto",
        Path("Proyecto"),
    ]
    for candidate in candidates:
        if (candidate / "train.csv").exists() and (candidate / "eval.csv").exists():
            return candidate

    if IN_KAGGLE:
        for train_path in Path("/kaggle/input").rglob("train.csv"):
            if (train_path.parent / "eval.csv").exists():
                return train_path.parent

    raise FileNotFoundError("No encontre train.csv y eval.csv.")


DATA_DIR = find_data_dir()
TRAIN_PATH = DATA_DIR / "train.csv"
EVAL_PATH = DATA_DIR / "eval.csv"
print("data:", DATA_DIR)

# %%
train_df = pd.read_csv(TRAIN_PATH)
eval_df = pd.read_csv(EVAL_PATH)

train_df["text"] = train_df["text"].fillna("").astype(str)
eval_df["text"] = eval_df["text"].fillna("").astype(str)

classes = [int(x) for x in sorted(train_df["decade"].unique())]
label2id = {c: i for i, c in enumerate(classes)}
id2label = {i: c for c, i in label2id.items()}
train_df["label"] = train_df["decade"].map(label2id)

print("train:", train_df.shape)
print("eval:", eval_df.shape)
print("classes:", len(classes), classes[0], classes[-1])

# %%
def augment_text(text, rng):
    pairs = {
        "s": "f",
        "f": "s",
        "u": "v",
        "v": "u",
        "i": "y",
        "y": "i",
        "c": "\u00e7",
    }

    chars = []
    for ch in str(text):
        if ch.isalpha() and rng.random() < 0.008:
            continue

        low = ch.lower()
        if low in pairs and rng.random() < 0.018:
            rep = pairs[low]
            ch = rep.upper() if ch.isupper() else rep

        chars.append(ch)

    return "".join(chars)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=384):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def softmax_np(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

train_base, val_df = train_test_split(
    train_df,
    test_size=VALID_SIZE,
    stratify=train_df["label"],
    random_state=SEED,
)

rng = random.Random(SEED)
aug_df = train_base.sample(frac=AUG_FRACTION, random_state=SEED).copy()
aug_df["text"] = aug_df["text"].apply(lambda x: augment_text(x, rng))

train_aug = pd.concat([train_base, aug_df], ignore_index=True)
train_aug = train_aug.sample(frac=1, random_state=SEED).reset_index(drop=True)

train_ds = TextDataset(train_aug["text"], train_aug["label"], tokenizer, MAX_LEN)
val_ds = TextDataset(val_df["text"], val_df["label"], tokenizer, MAX_LEN)
eval_ds = TextDataset(eval_df["text"], None, tokenizer, MAX_LEN)

print("train base:", train_base.shape)
print("train augmented:", train_aug.shape)
print("validacion:", val_df.shape)

# %%
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(classes),
    id2label={i: str(id2label[i]) for i in id2label},
    label2id={str(k): v for k, v in label2id.items()},
)

args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / f"runs_{RUN_NAME}"),
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    warmup_steps=500,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=100,
    report_to="none",
    fp16=torch.cuda.is_available(),
    seed=SEED,
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()
metrics = trainer.evaluate()
print("VALIDACION:", metrics)

with open(OUTPUT_DIR / f"metrics_{RUN_NAME}.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

val_pred = trainer.predict(val_ds)
val_labels = np.argmax(val_pred.predictions, axis=1)
print(
    classification_report(
        val_df["label"],
        val_labels,
        labels=list(range(len(classes))),
        target_names=[str(c) for c in classes],
        zero_division=0,
    )
)

# %%
def predict_and_save(trainer_obj, suffix):
    pred = trainer_obj.predict(eval_ds)
    logits = pred.predictions
    probs = softmax_np(logits)
    pred_labels = np.argmax(probs, axis=1)
    answers = [id2label[int(i)] for i in pred_labels]

    submission = pd.DataFrame({"id": eval_df["id"], "answer": answers})
    sub_path = OUTPUT_DIR / f"submission_{RUN_NAME}_{suffix}.csv"
    probs_path = OUTPUT_DIR / f"probs_{RUN_NAME}_{suffix}.npy"

    submission.to_csv(sub_path, index=False)
    np.save(probs_path, probs)
    print("submission:", sub_path)
    print("probs:", probs_path)
    return sub_path, probs_path


predict_and_save(trainer, "bestval")

# %%
# Ajuste final con 100% de train, guardando puntos intermedios.
rng = random.Random(SEED)
final_aug = train_df.sample(frac=AUG_FRACTION, random_state=SEED).copy()
final_aug["text"] = final_aug["text"].apply(lambda x: augment_text(x, rng))

train_full = pd.concat([train_df, final_aug], ignore_index=True)
train_full = train_full.sample(frac=1, random_state=SEED).reset_index(drop=True)
full_ds = TextDataset(train_full["text"], train_full["label"], tokenizer, MAX_LEN)

num_gpus = max(1, torch.cuda.device_count())
steps_per_epoch = math.ceil(len(full_ds) / (BATCH_SIZE * num_gpus * GRAD_ACCUM))
print("train full:", train_full.shape)
print("steps_per_epoch:", steps_per_epoch)

done_fraction = 0.0
for i, fraction in enumerate(FINAL_FRACTIONS, start=1):
    max_steps = max(1, int(round(steps_per_epoch * fraction)))
    done_fraction += fraction
    args_full = TrainingArguments(
        output_dir=str(OUTPUT_DIR / f"runs_{RUN_NAME}_full_{i}"),
        learning_rate=FINAL_LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=max_steps,
        weight_decay=0.01,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=100,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=SEED,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=trainer.model,
        args=args_full,
        train_dataset=full_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    suffix = f"full_{int(round(done_fraction * 100)):03d}"
    predict_and_save(trainer, suffix)

# %%
save_dir = OUTPUT_DIR / f"modelo_{RUN_NAME}_final"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

with open(save_dir / "label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "classes": classes,
            "label2id": {str(k): int(v) for k, v in label2id.items()},
            "id2label": {str(k): int(v) for k, v in id2label.items()},
        },
        f,
        indent=2,
    )

if SAVE_MODEL_ZIP:
    shutil.make_archive(str(save_dir), "zip", save_dir)

shutil.rmtree(OUTPUT_DIR / f"runs_{RUN_NAME}", ignore_errors=True)
for i in range(1, len(FINAL_FRACTIONS) + 1):
    shutil.rmtree(OUTPUT_DIR / f"runs_{RUN_NAME}_full_{i}", ignore_errors=True)

print("modelo:", save_dir)
