# %% [markdown]
# # Parte 2 - experimentos largos para Kaggle/Colab
#
# Este notebook esta preparado para correr en Kaggle en segundo plano.
# Detecta `train.csv` y `eval.csv` en `/kaggle/input`, guarda salidas en
# `/kaggle/working`, y tambien funciona en Colab si los CSV estan en Drive.
#
# Orden sugerido:
#
# Como BETO con `max_len=384` ya logro 0.27889 y XLM-R dio bajo, el primer
# objetivo es explotar variantes de BETO 384 antes de probar otros modelos:
#
# 1. `beto_384_trunc_s53`: BETO cased, 384 tokens, otra semilla.
# 2. `beto_uncased_384_s52`: BETO uncased, variante complementaria.
# 3. `bertin_roberta_384_s47`: BERTIN RoBERTa.
# 4. `mdeberta_v3_base_headtail_s48`: mDeBERTa v3, ultima por costo/riesgo.
#
# Notas:
# - `beto_512_headtail_s51` ya se probo y dio 0.27125.
# - `BSC-LT/roberta-base-bne` y `PlanTL-GOB-ES/roberta-base-bne` fallaron en
#   Kaggle/Hugging Face, asi que no se dejan por defecto.
#
# En Kaggle conviene correr un modelo por intento para que el notebook termine
# y deje outputs. Cada run genera `submission_*.csv` y `probs_*.npy`.

# %%
INSTALL_PACKAGES = True

if INSTALL_PACKAGES:
    import subprocess
    import sys

    # En Colab conviene no actualizar numpy/pandas/scikit-learn: vienen
    # compilados como conjunto y un upgrade parcial puede romper imports.
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--upgrade",
        "transformers>=4.45,<5",
        "accelerate",
        "sentencepiece",
        "protobuf<6",
    ])

# %%
from pathlib import Path
import gc
import inspect
import json
import os
import random
import re
import shutil
import unicodedata

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

IN_KAGGLE = Path("/kaggle/working").exists()
IN_COLAB = False
drive = None

if not IN_KAGGLE and Path("/var/colab/hostname").exists():
    try:
        from google.colab import drive

        IN_COLAB = True
    except Exception:
        IN_COLAB = False
        drive = None

MOUNT_DRIVE = IN_COLAB
if IN_COLAB and MOUNT_DRIVE and drive is not None:
    drive.mount("/content/drive")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)
if DEVICE == "cuda":
    print("gpu:", torch.cuda.get_device_name(0))

# %%
CANDIDATE_DIRS = [
    Path("/kaggle/input/datasets/jsmaldo/proyecto-archivos"),
    Path("/kaggle/input/proyecto-archivos"),
    Path.cwd(),
    Path.cwd() / "Proyecto",
    Path("/content"),
    Path("/content/Proyecto"),
    Path("/content/drive/MyDrive/Proyecto"),
    Path("/content/drive/MyDrive/proyecto"),
    Path("/content/drive/MyDrive"),
]

PROYECTO_DIR = None
for candidate in CANDIDATE_DIRS:
    if (candidate / "train.csv").exists() and (candidate / "eval.csv").exists():
        PROYECTO_DIR = candidate
        break

if PROYECTO_DIR is None and IN_KAGGLE:
    for train_path in Path("/kaggle/input").rglob("train.csv"):
        if (train_path.parent / "eval.csv").exists():
            PROYECTO_DIR = train_path.parent
            break

if PROYECTO_DIR is None:
    raise FileNotFoundError(
        "No encontre train.csv y eval.csv. En Kaggle agrega el dataset de archivos; "
        "en Colab sube ambos a /content o a Drive/Proyecto."
    )

if IN_KAGGLE:
    OUTPUT_ROOT = Path("/kaggle/working")
elif IN_COLAB:
    OUTPUT_ROOT = Path("/content/drive/MyDrive/proyecto_parte2_runs")
else:
    OUTPUT_ROOT = PROYECTO_DIR / "runs_parte2_colab"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(PROYECTO_DIR / "train.csv")
eval_df = pd.read_csv(PROYECTO_DIR / "eval.csv")

train_df["text"] = train_df["text"].fillna("").astype(str)
eval_df["text"] = eval_df["text"].fillna("").astype(str)

classes = [int(x) for x in sorted(train_df["decade"].unique())]
label2id = {c: i for i, c in enumerate(classes)}
id2label = {i: c for c, i in label2id.items()}
train_df["label"] = train_df["decade"].map(label2id).astype(int)

print("proyecto:", PROYECTO_DIR)
print("salidas:", OUTPUT_ROOT)
print("train:", train_df.shape)
print("eval:", eval_df.shape)
print("clases:", len(classes), classes[0], classes[-1])
print(train_df["text"].str.len().describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

# %%
EXPERIMENTS = [
    {
        "run_name": "beto_384_trunc_s53",
        "model_name": "dccuchile/bert-base-spanish-wwm-cased",
        "seed": 53,
        "max_len": 384,
        "valid_size": 0.12,
        "epochs": 4,
        "lr": 2e-5,
        "batch_train": 4,
        "batch_eval": 8,
        "grad_accum": 4,
        "warmup_ratio": 0.07,
        "weight_decay": 0.01,
        "aug_fraction": 0.15,
        "head_tail": False,
        "gradient_checkpointing": True,
    },
    {
        "run_name": "beto_uncased_384_s52",
        "model_name": "dccuchile/bert-base-spanish-wwm-uncased",
        "seed": 52,
        "max_len": 384,
        "valid_size": 0.12,
        "epochs": 4,
        "lr": 2e-5,
        "batch_train": 4,
        "batch_eval": 8,
        "grad_accum": 4,
        "warmup_ratio": 0.08,
        "weight_decay": 0.01,
        "aug_fraction": 0.18,
        "head_tail": True,
        "gradient_checkpointing": True,
    },
    {
        "run_name": "bertin_roberta_384_s47",
        "model_name": "bertin-project/bertin-roberta-base-spanish",
        "seed": 47,
        "max_len": 384,
        "valid_size": 0.12,
        "epochs": 4,
        "lr": 1.5e-5,
        "batch_train": 4,
        "batch_eval": 8,
        "grad_accum": 4,
        "warmup_ratio": 0.10,
        "weight_decay": 0.01,
        "aug_fraction": 0.20,
        "head_tail": True,
        "gradient_checkpointing": True,
    },
    {
        "run_name": "mdeberta_v3_base_headtail_s48",
        "model_name": "microsoft/mdeberta-v3-base",
        "seed": 48,
        "max_len": 384,
        "valid_size": 0.12,
        "epochs": 3,
        "lr": 1.5e-5,
        "batch_train": 4,
        "batch_eval": 8,
        "grad_accum": 4,
        "warmup_ratio": 0.10,
        "weight_decay": 0.01,
        "aug_fraction": 0.15,
        "head_tail": True,
        "gradient_checkpointing": True,
    },
]

# Siguiente intento recomendado: BETO 384 con otra semilla.
RUN_INDEX = 0

# Un modelo por ejecucion: reduce el riesgo de perder outputs por timeout/fallos.
RUN_ALL = False

# Para el run final de entrega puedes activar el zip.
SAVE_MODEL = True
ZIP_MODEL = False
REMOVE_CHECKPOINTS_AFTER_RUN = True

# Ajuste corto despues de escoger el mejor checkpoint por validacion.
# Usa 100% de train.csv antes de predecir eval.csv.
FINAL_FULL_TUNE = True
FINAL_FULL_EPOCHS = 1
FINAL_FULL_LR = 8e-6

# %%
def clean_spaces(text):
    text = re.sub(r"-\s*\n\s*", "", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_accents_lite(ch):
    decomposed = unicodedata.normalize("NFD", ch)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped if stripped else ch


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
    for ch in clean_spaces(text):
        if ch.isalpha() and rng.random() < 0.010:
            continue

        low = ch.lower()
        if low in pairs and rng.random() < 0.020:
            repl = pairs[low]
            ch = repl.upper() if ch.isupper() else repl
        elif ch.isalpha() and rng.random() < 0.006:
            ch = strip_accents_lite(ch)

        chars.append(ch)

    words = "".join(chars).split()
    for i in range(len(words) - 1):
        if rng.random() < 0.006:
            words[i], words[i + 1] = words[i + 1], words[i]

    return " ".join(words)


def make_split_and_augmented_data(cfg):
    seed = cfg["seed"]
    train_base, val_df = train_test_split(
        train_df,
        test_size=cfg["valid_size"],
        stratify=train_df["label"],
        random_state=seed,
    )

    rng = random.Random(seed)
    aug_fraction = cfg.get("aug_fraction", 0.0)
    if aug_fraction > 0:
        aug_df = train_base.sample(frac=aug_fraction, random_state=seed).copy()
        aug_df["text"] = aug_df["text"].apply(lambda x: augment_text(x, rng))
        train_aug = pd.concat([train_base, aug_df], ignore_index=True)
    else:
        train_aug = train_base.copy()

    train_aug["text"] = train_aug["text"].map(clean_spaces)
    val_df = val_df.copy()
    val_df["text"] = val_df["text"].map(clean_spaces)

    train_aug = train_aug.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print("train base:", train_base.shape)
    print("train augmented:", train_aug.shape)
    print("validacion:", val_df.shape)
    return train_aug, val_df


def make_full_augmented_data(cfg):
    seed = cfg["seed"]
    full_df = train_df.copy()

    rng = random.Random(seed)
    aug_fraction = cfg.get("aug_fraction", 0.0)
    if aug_fraction > 0:
        aug_df = full_df.sample(frac=aug_fraction, random_state=seed).copy()
        aug_df["text"] = aug_df["text"].apply(lambda x: augment_text(x, rng))
        full_aug = pd.concat([full_df, aug_df], ignore_index=True)
    else:
        full_aug = full_df

    full_aug["text"] = full_aug["text"].map(clean_spaces)
    full_aug = full_aug.sample(frac=1, random_state=seed).reset_index(drop=True)
    print("train final full:", full_aug.shape)
    return full_aug

# %%
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = None if labels is None else list(labels)

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(values[idx]) for key, values in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def encode_texts(texts, tokenizer, max_len, head_tail=True):
    output = {}
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    token_budget = max_len - special_tokens

    for text in texts:
        if head_tail:
            ids = tokenizer.encode(
                str(text),
                add_special_tokens=False,
                truncation=False,
            )
            if len(ids) > token_budget:
                head_len = token_budget // 2
                tail_len = token_budget - head_len
                ids = ids[:head_len] + ids[-tail_len:]

            encoded = tokenizer.prepare_for_model(
                ids,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
            )
        else:
            encoded = tokenizer(
                str(text),
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
            )

        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key in encoded:
                output.setdefault(key, []).append(encoded[key])

    return output


def build_datasets(cfg, tokenizer, train_aug, val_df):
    print("tokenizando train...")
    train_enc = encode_texts(
        train_aug["text"],
        tokenizer,
        cfg["max_len"],
        head_tail=cfg.get("head_tail", True),
    )
    print("tokenizando validacion...")
    val_enc = encode_texts(
        val_df["text"],
        tokenizer,
        cfg["max_len"],
        head_tail=cfg.get("head_tail", True),
    )

    train_ds = TextDataset(train_enc, train_aug["label"])
    val_ds = TextDataset(val_enc, val_df["label"])
    return train_ds, val_ds

# %%
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def make_training_args(cfg, run_dir):
    sig = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {
        "output_dir": str(run_dir / "checkpoints"),
        "learning_rate": cfg["lr"],
        "per_device_train_batch_size": cfg["batch_train"],
        "per_device_eval_batch_size": cfg["batch_eval"],
        "gradient_accumulation_steps": cfg["grad_accum"],
        "num_train_epochs": cfg["epochs"],
        "weight_decay": cfg["weight_decay"],
        "warmup_ratio": cfg["warmup_ratio"],
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "logging_steps": 100,
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
        "seed": cfg["seed"],
    }

    if "eval_strategy" in sig:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    if "optim" in sig:
        kwargs["optim"] = "adamw_torch"
    if "gradient_checkpointing" in sig:
        kwargs["gradient_checkpointing"] = cfg.get("gradient_checkpointing", False)
    if "dataloader_num_workers" in sig:
        kwargs["dataloader_num_workers"] = 2

    return TrainingArguments(**kwargs)


def make_trainer(model, tokenizer, args, train_ds, val_ds=None):
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_ds,
    }
    if val_ds is not None:
        trainer_kwargs["eval_dataset"] = val_ds
        trainer_kwargs["compute_metrics"] = compute_metrics
    sig = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in sig:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    return Trainer(**trainer_kwargs)


def make_final_training_args(cfg, run_dir):
    sig = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {
        "output_dir": str(run_dir / "checkpoints_full"),
        "learning_rate": FINAL_FULL_LR,
        "per_device_train_batch_size": cfg["batch_train"],
        "per_device_eval_batch_size": cfg["batch_eval"],
        "gradient_accumulation_steps": cfg["grad_accum"],
        "num_train_epochs": FINAL_FULL_EPOCHS,
        "weight_decay": cfg["weight_decay"],
        "save_strategy": "no",
        "logging_steps": 100,
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
        "seed": cfg["seed"],
    }

    if "eval_strategy" in sig:
        kwargs["eval_strategy"] = "no"
    else:
        kwargs["evaluation_strategy"] = "no"
    if "optim" in sig:
        kwargs["optim"] = "adamw_torch"
    if "gradient_checkpointing" in sig:
        kwargs["gradient_checkpointing"] = cfg.get("gradient_checkpointing", False)
    if "dataloader_num_workers" in sig:
        kwargs["dataloader_num_workers"] = 2

    return TrainingArguments(**kwargs)


def softmax_np(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)

# %%
def train_one(cfg):
    print(json.dumps(cfg, indent=2))
    seed = cfg["seed"]
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_dir = OUTPUT_ROOT / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    train_aug, val_df = make_split_and_augmented_data(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    train_ds, val_ds = build_datasets(cfg, tokenizer, train_aug, val_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=len(classes),
        id2label={i: str(id2label[i]) for i in id2label},
        label2id={str(k): v for k, v in label2id.items()},
    )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    args = make_training_args(cfg, run_dir)
    trainer = make_trainer(model, tokenizer, args, train_ds, val_ds)

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    with open(run_dir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    val_pred = trainer.predict(val_ds)
    val_pred_labels = np.argmax(val_pred.predictions, axis=1)
    np.save(run_dir / "val_logits.npy", val_pred.predictions)

    report = classification_report(
        val_df["label"],
        val_pred_labels,
        labels=list(range(len(classes))),
        target_names=[str(c) for c in classes],
        zero_division=0,
    )
    print(report)
    (run_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    if FINAL_FULL_TUNE:
        print("ajuste final con 100% de train...")
        full_aug = make_full_augmented_data(cfg)
        full_enc = encode_texts(
            full_aug["text"],
            tokenizer,
            cfg["max_len"],
            head_tail=cfg.get("head_tail", True),
        )
        full_ds = TextDataset(full_enc, full_aug["label"])
        args_full = make_final_training_args(cfg, run_dir)
        trainer = make_trainer(trainer.model, tokenizer, args_full, full_ds)
        trainer.train()

    eval_clean = eval_df.copy()
    eval_clean["text"] = eval_clean["text"].map(clean_spaces)
    print("tokenizando eval...")
    eval_enc = encode_texts(
        eval_clean["text"],
        tokenizer,
        cfg["max_len"],
        head_tail=cfg.get("head_tail", True),
    )
    eval_ds = TextDataset(eval_enc)

    eval_pred = trainer.predict(eval_ds)
    eval_logits = eval_pred.predictions
    eval_probs = softmax_np(eval_logits)
    np.save(run_dir / "eval_logits.npy", eval_logits)
    np.save(run_dir / "eval_probs.npy", eval_probs)
    np.save(OUTPUT_ROOT / f"probs_{cfg['run_name']}.npy", eval_probs)

    pred_idx = np.argmax(eval_logits, axis=1)
    answers = [id2label[int(i)] for i in pred_idx]
    submission = pd.DataFrame({"id": eval_df["id"], "answer": answers})
    submission_path = run_dir / f"submission_{cfg['run_name']}.csv"
    submission_root_path = OUTPUT_ROOT / f"submission_{cfg['run_name']}.csv"
    submission.to_csv(submission_path, index=False)
    submission.to_csv(submission_root_path, index=False)

    mapping = {
        "classes": classes,
        "label2id": {str(k): int(v) for k, v in label2id.items()},
        "id2label": {str(k): int(v) for k, v in id2label.items()},
    }
    with open(run_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    if SAVE_MODEL:
        model_dir = run_dir / "model"
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)
        if ZIP_MODEL:
            shutil.make_archive(str(model_dir), "zip", model_dir)

    if REMOVE_CHECKPOINTS_AFTER_RUN:
        shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)
        shutil.rmtree(run_dir / "checkpoints_full", ignore_errors=True)

    print("submission:", submission_path)
    print("submission root:", submission_root_path)
    print("probabilidades:", OUTPUT_ROOT / f"probs_{cfg['run_name']}.npy")
    return run_dir

# %%
selected_experiments = EXPERIMENTS if RUN_ALL else [EXPERIMENTS[RUN_INDEX]]
finished_runs = []

for cfg in selected_experiments:
    finished_runs.append(train_one(cfg))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

finished_runs

# %% [markdown]
# ## Ensamble por probabilidades
#
# Ejecuta esta celda despues de tener dos o mas carpetas con `eval_logits.npy`.
# Si solo tienes un run, no hace falta: sube el `submission_*.csv` de ese run.

# %%
ENSEMBLE_RUN_NAMES = []  # vacio = usa todos los runs con eval_probs/eval_logits
ENSEMBLE_WEIGHTED_BY_VAL = False
INCLUDE_EXTERNAL_BETO_384 = True
INCLUDE_EXTERNAL_XLMR_384 = False  # dio 0.20534 publico; mejor dejarlo fuera

MANUAL_ENSEMBLE_WEIGHTS = {
    "probs_beto_384.npy": 1.35,
    "probs_xlmr_384.npy": 0.20,
}

run_dirs = [
    p
    for p in sorted(OUTPUT_ROOT.iterdir())
    if p.is_dir() and ((p / "eval_probs.npy").exists() or (p / "eval_logits.npy").exists())
]

if ENSEMBLE_RUN_NAMES:
    wanted = set(ENSEMBLE_RUN_NAMES)
    run_dirs = [p for p in run_dirs if p.name in wanted]

probs = []
weights = []
names = []

for run_dir in run_dirs:
    if (run_dir / "eval_probs.npy").exists():
        prob = np.load(run_dir / "eval_probs.npy")
    else:
        prob = softmax_np(np.load(run_dir / "eval_logits.npy"))

    if prob.shape != (len(eval_df), len(classes)):
        print("Saltando por forma inesperada:", run_dir, prob.shape)
        continue

    weight = 1.0
    metrics_path = run_dir / "val_metrics.json"
    if ENSEMBLE_WEIGHTED_BY_VAL and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        weight = float(metrics.get("eval_accuracy", metrics.get("accuracy", 1.0)))

    probs.append(prob)
    weights.append(weight)
    names.append(run_dir.name)
    print("run", run_dir.name, "weight", weight)

external_prob_files = []
if IN_KAGGLE:
    external_prob_files.extend(Path("/kaggle/input").rglob("probs_*.npy"))

for prob_path in sorted(set(external_prob_files), key=str):
    if prob_path.name == "probs_beto_384.npy" and not INCLUDE_EXTERNAL_BETO_384:
        continue
    if prob_path.name == "probs_xlmr_384.npy" and not INCLUDE_EXTERNAL_XLMR_384:
        continue
    if prob_path.name.startswith("probs_xlmr") and not INCLUDE_EXTERNAL_XLMR_384:
        continue

    prob = np.load(prob_path)
    if prob.shape != (len(eval_df), len(classes)):
        print("Saltando externo por forma inesperada:", prob_path, prob.shape)
        continue

    weight = MANUAL_ENSEMBLE_WEIGHTS.get(prob_path.name, 1.0)
    probs.append(prob)
    weights.append(weight)
    names.append(str(prob_path))
    print("externo", prob_path, "weight", weight)

if len(probs) < 2:
    print("Necesitas al menos dos fuentes para ensamblar. Fuentes:", names)
else:
    stacked = np.stack(probs, axis=0)
    weights_arr = np.array(weights, dtype=float).reshape(-1, 1, 1)
    avg_probs = (stacked * weights_arr).sum(axis=0) / weights_arr.sum()

    pred_idx = np.argmax(avg_probs, axis=1)
    answers = [id2label[int(i)] for i in pred_idx]
    ensemble = pd.DataFrame({"id": eval_df["id"], "answer": answers})

    ensemble_name = "submission_ensemble_probs.csv"
    ensemble_path = OUTPUT_ROOT / ensemble_name
    ensemble.to_csv(ensemble_path, index=False)
    print("ensemble:", ensemble_path)
    print("fuentes:")
    for name, weight in zip(names, weights):
        print("-", weight, name)
    display(ensemble.head())

# %%
model_refs = "\n".join(
    f"- `{cfg['model_name']}`: https://huggingface.co/{cfg['model_name']}"
    for cfg in EXPERIMENTS
)

references = f"""# Referencias y datos externos - Parte 2

No se usaron datos externos adicionales para entrenar el clasificador.
La aumentacion fue sintetica y se genero exclusivamente a partir de `train.csv`,
sin usar `eval.csv` para entrenamiento.

## Modelos preentrenados candidatos

{model_refs}

## Aumentacion sintetica

Se aplicaron transformaciones ligeras sobre una muestra del entrenamiento:

1. Limpieza de saltos de linea y guiones de corte.
2. Eliminacion aleatoria muy baja de caracteres alfabeticos.
3. Sustituciones OCR/ortografia historica: `s/f`, `u/v`, `i/y`, `c/cedilla`.
4. Perdida ocasional de acentos.
5. Intercambio ocasional de palabras consecutivas.
"""

ref_path = OUTPUT_ROOT / "referencias_datos_parte2.md"
ref_path.write_text(references, encoding="utf-8")
print(ref_path)
