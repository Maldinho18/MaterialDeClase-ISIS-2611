# %% [markdown]
# # Postproceso rapido para BETO 384
#
# Este notebook no entrena. Usa las probabilidades del mejor run conocido
# (`probs_beto_384.npy`, score publico 0.27889) y genera submissions nuevas con
# correccion de prior y rebalanceo de clases.
#
# En Kaggle agrega como input:
#
# 1. El dataset con `train.csv` y `eval.csv`.
# 2. El output del notebook `beto-y-xmlr-beto0-27`, que contiene
#    `probs_beto_384.npy`.
#
# El objetivo es obtener candidatos en minutos, no en horas.

# %%
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

IN_KAGGLE = Path("/kaggle/working").exists()
OUTPUT_DIR = Path("/kaggle/working") if IN_KAGGLE else Path("Proyecto/postproceso_beto384")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def find_prob_file(name="probs_beto_384.npy"):
    candidates = []
    if IN_KAGGLE:
        candidates.extend(Path("/kaggle/input").rglob(name))
        candidates.extend(Path("/kaggle/working").rglob(name))
    candidates.extend(Path.cwd().rglob(name))

    candidates = sorted(set(candidates), key=lambda p: (len(str(p)), str(p)))
    if not candidates:
        raise FileNotFoundError(
            f"No encontre {name}. Agrega como input el output del notebook que genero BETO 384."
        )
    return candidates[0]


DATA_DIR = find_data_dir()
PROBS_PATH = find_prob_file("probs_beto_384.npy")

print("data:", DATA_DIR)
print("probs:", PROBS_PATH)
print("output:", OUTPUT_DIR)

# %%
train_df = pd.read_csv(DATA_DIR / "train.csv")
eval_df = pd.read_csv(DATA_DIR / "eval.csv")

classes = np.array([int(x) for x in sorted(train_df["decade"].unique())])
class_to_pos = {c: i for i, c in enumerate(classes)}
probs = np.load(PROBS_PATH)

if probs.shape != (len(eval_df), len(classes)):
    raise ValueError(f"Forma inesperada en probs: {probs.shape}. Esperaba {(len(eval_df), len(classes))}.")

train_counts = train_df["decade"].value_counts().reindex(classes).to_numpy()
train_prior = train_counts / train_counts.sum()
uniform_prior = np.ones(len(classes)) / len(classes)

print("eval:", eval_df.shape)
print("classes:", len(classes), classes[0], classes[-1])
print("probs:", probs.shape)

# %%
def save_submission(labels_pos, name):
    answers = classes[np.asarray(labels_pos, dtype=int)]
    sub = pd.DataFrame({"id": eval_df["id"], "answer": answers})
    path = OUTPUT_DIR / name
    sub.to_csv(path, index=False)
    counts = sub["answer"].value_counts().reindex(classes, fill_value=0)
    print(name, "->", path)
    print("  min/max counts:", int(counts.min()), int(counts.max()))
    return path


def rounded_counts(prior, total):
    raw = prior * total
    counts = np.floor(raw).astype(int)
    missing = total - counts.sum()
    if missing > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:missing]] += 1
    elif missing < 0:
        order = np.argsort(raw - counts)
        for idx in order[: -missing]:
            counts[idx] -= 1
    return counts


def prior_corrected_labels(probs, target_prior, alpha):
    pred_prior = probs.mean(axis=0)
    factors = (target_prior / np.clip(pred_prior, 1e-12, None)) ** alpha
    corrected = probs * factors[None, :]
    corrected = corrected / corrected.sum(axis=1, keepdims=True)
    return corrected.argmax(axis=1)


def rebalance_to_target(probs, target_counts, prior=None, prior_strength=0.0):
    scores = np.log(np.clip(probs, 1e-12, 1.0))
    if prior is not None and prior_strength:
        scores = scores + prior_strength * np.log(np.clip(prior, 1e-12, 1.0))[None, :]

    labels = scores.argmax(axis=1).astype(int)
    counts = np.bincount(labels, minlength=len(classes)).astype(int)
    target_counts = np.asarray(target_counts, dtype=int)

    max_moves = int(np.abs(counts - target_counts).sum() // 2 + len(labels))
    moves = 0

    while moves < max_moves:
        over = np.where(counts > target_counts)[0]
        under = np.where(counts < target_counts)[0]
        if len(over) == 0 or len(under) == 0:
            break

        best = None
        for cls in over:
            idx = np.where(labels == cls)[0]
            if len(idx) == 0:
                continue
            alt_scores = scores[np.ix_(idx, under)]
            best_alt_pos = alt_scores.argmax(axis=1)
            best_alt_cls = under[best_alt_pos]
            loss = scores[idx, cls] - alt_scores[np.arange(len(idx)), best_alt_pos]
            local = int(np.argmin(loss))
            candidate = (float(loss[local]), int(idx[local]), int(cls), int(best_alt_cls[local]))
            if best is None or candidate[0] < best[0]:
                best = candidate

        if best is None:
            break

        _, row, old_cls, new_cls = best
        labels[row] = new_cls
        counts[old_cls] -= 1
        counts[new_cls] += 1
        moves += 1

    return labels, counts

# %%
# Sanity check: debe reproducir la submission del BETO 384 original.
base_labels = probs.argmax(axis=1)
save_submission(base_labels, "submission_beto384_argmax_from_probs.csv")

target_train = rounded_counts(train_prior, len(eval_df))
target_uniform = rounded_counts(uniform_prior, len(eval_df))
base_counts = np.bincount(base_labels, minlength=len(classes))

summary = {
    "base_counts_min": int(base_counts.min()),
    "base_counts_max": int(base_counts.max()),
    "train_target_min": int(target_train.min()),
    "train_target_max": int(target_train.max()),
    "uniform_target_min": int(target_uniform.min()),
    "uniform_target_max": int(target_uniform.max()),
}
print(json.dumps(summary, indent=2))

# %%
# 1) Correccion suave de prior. Empieza submitiendo alpha 0.25 y 0.50.
for alpha in [0.15, 0.25, 0.35, 0.50, 0.75, 1.00]:
    labels = prior_corrected_labels(probs, train_prior, alpha)
    save_submission(labels, f"submission_beto384_prior_train_a{alpha:.2f}.csv")

for alpha in [0.15, 0.25, 0.35, 0.50]:
    labels = prior_corrected_labels(probs, uniform_prior, alpha)
    save_submission(labels, f"submission_beto384_prior_uniform_a{alpha:.2f}.csv")

# %%
# 2) Rebalanceo duro de conteos. Es mas agresivo; prueba primero blend 0.25.
for alpha in [0.25, 0.50, 0.75, 1.00]:
    blended = (1 - alpha) * base_counts + alpha * target_train
    target = rounded_counts(blended / blended.sum(), len(eval_df))
    labels, counts = rebalance_to_target(probs, target)
    save_submission(labels, f"submission_beto384_quota_train_a{alpha:.2f}.csv")

for alpha in [0.25, 0.50]:
    blended = (1 - alpha) * base_counts + alpha * target_uniform
    target = rounded_counts(blended / blended.sum(), len(eval_df))
    labels, counts = rebalance_to_target(probs, target)
    save_submission(labels, f"submission_beto384_quota_uniform_a{alpha:.2f}.csv")

# %%
print("Candidatos recomendados para probar primero:")
print("1. submission_beto384_prior_train_a0.25.csv")
print("2. submission_beto384_prior_train_a0.35.csv")
print("3. submission_beto384_quota_train_a0.25.csv")
print("4. submission_beto384_prior_uniform_a0.25.csv")
