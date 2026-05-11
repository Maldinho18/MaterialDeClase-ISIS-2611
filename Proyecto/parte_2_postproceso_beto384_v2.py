# %% [markdown]
# # Postproceso BETO 384 v2
#
# Usa `probs_beto_384.npy` del run que dio 0.27889 y genera una segunda tanda
# de candidatos sin entrenar. La primera tanda ya subio a 0.27984, asi que aqui
# exploramos cambios mas finos y menos agresivos.
#
# Corre sin GPU. En Kaggle agrega como inputs:
#
# 1. Dataset con `train.csv` y `eval.csv`.
# 2. Output del notebook `beto-y-xmlr-beto0-27`, que contiene
#    `probs_beto_384.npy`.

# %%
from pathlib import Path
import json

import numpy as np
import pandas as pd

IN_KAGGLE = Path("/kaggle/working").exists()
OUTPUT_DIR = Path("/kaggle/working") if IN_KAGGLE else Path("Proyecto/postproceso_beto384_v2")
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
            f"No encontre {name}. Agrega como input el output del notebook BETO 384."
        )
    return candidates[0]


DATA_DIR = find_data_dir()
PROBS_PATH = find_prob_file()

print("data:", DATA_DIR)
print("probs:", PROBS_PATH)
print("output:", OUTPUT_DIR)

# %%
train_df = pd.read_csv(DATA_DIR / "train.csv")
eval_df = pd.read_csv(DATA_DIR / "eval.csv")

classes = np.array([int(x) for x in sorted(train_df["decade"].unique())])
probs = np.load(PROBS_PATH).astype(np.float64)

if probs.shape != (len(eval_df), len(classes)):
    raise ValueError(f"Forma inesperada: {probs.shape}. Esperada: {(len(eval_df), len(classes))}")

train_counts = train_df["decade"].value_counts().reindex(classes).to_numpy()
train_prior = train_counts / train_counts.sum()
uniform_prior = np.ones(len(classes)) / len(classes)

base_labels = probs.argmax(axis=1)
base_counts = np.bincount(base_labels, minlength=len(classes))

sorted_probs = np.sort(probs, axis=1)
base_margin = sorted_probs[:, -1] - sorted_probs[:, -2]

print("eval:", eval_df.shape)
print("classes:", len(classes), classes[0], classes[-1])
print("base count min/max:", int(base_counts.min()), int(base_counts.max()))
print("margin quantiles:", np.quantile(base_margin, [0.1, 0.25, 0.5, 0.75, 0.9]))

# %%
manifest = []
saved = {}


def normalize(p):
    return p / p.sum(axis=1, keepdims=True)


def temperature_scale(p, temperature):
    power = 1.0 / temperature
    return normalize(np.clip(p, 1e-12, 1.0) ** power)


def prior_correct(p, target_prior, alpha):
    pred_prior = p.mean(axis=0)
    factors = (target_prior / np.clip(pred_prior, 1e-12, None)) ** alpha
    return normalize(p * factors[None, :])


def neighbor_smooth(p, strength):
    left = np.zeros_like(p)
    right = np.zeros_like(p)
    left[:, 1:] = p[:, :-1]
    right[:, :-1] = p[:, 1:]
    smoothed = (1 - strength) * p + strength * 0.5 * (left + right)
    return normalize(smoothed)


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


def rebalance_to_target(p, target_counts):
    scores = np.log(np.clip(p, 1e-12, 1.0))
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

    return labels


def save_labels(labels, name, desc):
    labels = np.asarray(labels, dtype=int)
    answers = classes[labels]
    sub = pd.DataFrame({"id": eval_df["id"], "answer": answers})
    path = OUTPUT_DIR / f"{name}.csv"
    sub.to_csv(path, index=False)

    counts = np.bincount(labels, minlength=len(classes))
    changes = int((labels != base_labels).sum())
    key = tuple(labels.tolist())
    duplicate_of = saved.get(key)
    if duplicate_of is None:
        saved[key] = name
    manifest.append(
        {
            "name": name,
            "file": str(path),
            "desc": desc,
            "changes_vs_base": changes,
            "count_min": int(counts.min()),
            "count_max": int(counts.max()),
            "duplicate_of": duplicate_of or "",
        }
    )
    print(name, "changes", changes, "count min/max", int(counts.min()), int(counts.max()), "dup", duplicate_of or "-")
    return path

# %%
# Base para verificar que las probabilidades corresponden al BETO 384 original.
save_labels(base_labels, "submission_v2_beto384_base_argmax", "argmax original desde probs_beto_384")

# %%
# A. Prior correction fino alrededor de lo que ya subio a 0.27984.
for alpha in [0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.42]:
    corrected = prior_correct(probs, train_prior, alpha)
    save_labels(
        corrected.argmax(axis=1),
        f"submission_v2_prior_train_a{alpha:.2f}",
        f"prior train alpha={alpha:.2f}",
    )

for alpha in [0.18, 0.22, 0.25, 0.28, 0.32, 0.38]:
    corrected = prior_correct(probs, uniform_prior, alpha)
    save_labels(
        corrected.argmax(axis=1),
        f"submission_v2_prior_uniform_a{alpha:.2f}",
        f"prior uniform alpha={alpha:.2f}",
    )

# %%
# B. Temperatura + prior. La temperatura cambia solo los casos ambiguos.
for temperature in [0.85, 0.95, 1.05, 1.15, 1.30]:
    p_temp = temperature_scale(probs, temperature)
    for alpha in [0.20, 0.25, 0.30, 0.35]:
        corrected = prior_correct(p_temp, train_prior, alpha)
        save_labels(
            corrected.argmax(axis=1),
            f"submission_v2_temp{temperature:.2f}_train_a{alpha:.2f}",
            f"temperature={temperature:.2f}, prior train alpha={alpha:.2f}",
        )

# %%
# C. Cambios selectivos: aplica prior correction solo donde BETO estaba menos seguro.
for alpha in [0.25, 0.30, 0.35]:
    corrected = prior_correct(probs, train_prior, alpha)
    corr_labels = corrected.argmax(axis=1)
    for q in [0.20, 0.30, 0.40, 0.50, 0.60]:
        labels = base_labels.copy()
        threshold = np.quantile(base_margin, q)
        mask = (base_margin <= threshold) & (corr_labels != base_labels)
        labels[mask] = corr_labels[mask]
        save_labels(
            labels,
            f"submission_v2_select_q{q:.2f}_train_a{alpha:.2f}",
            f"selective prior train alpha={alpha:.2f}, margin quantile={q:.2f}",
        )

# %%
# D. Suavizado ordinal leve: reparte un poco de probabilidad entre decadas vecinas.
for strength in [0.03, 0.05, 0.08, 0.10]:
    p_smooth = neighbor_smooth(probs, strength)
    for alpha in [0.20, 0.25, 0.30]:
        corrected = prior_correct(p_smooth, train_prior, alpha)
        save_labels(
            corrected.argmax(axis=1),
            f"submission_v2_smooth{strength:.2f}_train_a{alpha:.2f}",
            f"neighbor smooth={strength:.2f}, prior train alpha={alpha:.2f}",
        )

# %%
# E. Cuotas suaves. Ya vimos que quota_train a0.25 no subio, pero uniform a0.25 si.
for target_name, target_prior in [("train", train_prior), ("uniform", uniform_prior)]:
    target_full = rounded_counts(target_prior, len(eval_df))
    for alpha in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        blended = (1 - alpha) * base_counts + alpha * target_full
        target = rounded_counts(blended / blended.sum(), len(eval_df))
        labels = rebalance_to_target(probs, target)
        save_labels(
            labels,
            f"submission_v2_quota_{target_name}_a{alpha:.2f}",
            f"quota {target_name} blend alpha={alpha:.2f}",
        )

# %%
manifest_df = pd.DataFrame(manifest)
manifest_path = OUTPUT_DIR / "manifest_postproceso_beto384_v2.csv"
manifest_df.to_csv(manifest_path, index=False)

unique_manifest = manifest_df[manifest_df["duplicate_of"].eq("")].copy()
unique_manifest["abs_changes_from_160"] = (unique_manifest["changes_vs_base"] - 160).abs()
recommended = unique_manifest.sort_values(
    ["abs_changes_from_160", "changes_vs_base", "name"]
).head(12)

recommended_path = OUTPUT_DIR / "recommended_postproceso_beto384_v2.csv"
recommended.to_csv(recommended_path, index=False)

print("manifest:", manifest_path)
print("recommended:", recommended_path)
print("Recomendados para enviar primero:")
print(recommended[["name", "changes_vs_base", "count_min", "count_max", "desc"]].to_string(index=False))
