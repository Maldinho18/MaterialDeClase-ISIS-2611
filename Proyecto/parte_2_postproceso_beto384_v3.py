# %% [markdown]
# # Postproceso BETO 384 v3 - siguiente tanda
#
# La tanda v2 ya subio a 0.27984. Este notebook genera menos archivos que v2,
# enfocados en variantes mas diversas/agresivas para no gastar submissions en
# duplicados cercanos.
#
# No usa GPU. Requiere `train.csv`, `eval.csv` y `probs_beto_384.npy`.

# %%
from pathlib import Path
import json

import numpy as np
import pandas as pd

IN_KAGGLE = Path("/kaggle/working").exists()
OUTPUT_DIR = Path("/kaggle/working") if IN_KAGGLE else Path("Proyecto/postproceso_beto384_v3")
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
        raise FileNotFoundError(f"No encontre {name}.")
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

train_counts = train_df["decade"].value_counts().reindex(classes).to_numpy()
train_prior = train_counts / train_counts.sum()
uniform_prior = np.ones(len(classes)) / len(classes)
base_labels = probs.argmax(axis=1)
base_counts = np.bincount(base_labels, minlength=len(classes))

print("probs:", probs.shape)
print("base counts min/max:", int(base_counts.min()), int(base_counts.max()))

# %%
def normalize(p):
    return p / p.sum(axis=1, keepdims=True)


def temperature_scale(p, temperature):
    return normalize(np.clip(p, 1e-12, 1.0) ** (1.0 / temperature))


def prior_correct(p, target_prior, alpha):
    pred_prior = p.mean(axis=0)
    factors = (target_prior / np.clip(pred_prior, 1e-12, None)) ** alpha
    return normalize(p * factors[None, :])


def neighbor_smooth(p, strength):
    left = np.zeros_like(p)
    right = np.zeros_like(p)
    left[:, 1:] = p[:, :-1]
    right[:, :-1] = p[:, 1:]
    return normalize((1 - strength) * p + strength * 0.5 * (left + right))


def rounded_counts(prior, total):
    raw = prior * total
    counts = np.floor(raw).astype(int)
    missing = total - counts.sum()
    if missing > 0:
        counts[np.argsort(-(raw - counts))[:missing]] += 1
    elif missing < 0:
        for idx in np.argsort(raw - counts)[: -missing]:
            counts[idx] -= 1
    return counts


def rebalance_to_target(p, target_counts):
    scores = np.log(np.clip(p, 1e-12, 1.0))
    labels = scores.argmax(axis=1).astype(int)
    counts = np.bincount(labels, minlength=len(classes)).astype(int)
    target_counts = np.asarray(target_counts, dtype=int)

    max_moves = int(np.abs(counts - target_counts).sum() // 2 + len(labels))
    for _ in range(max_moves):
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
    return labels


manifest = []
seen = {}


def save(labels, name, desc, priority):
    labels = np.asarray(labels, dtype=int)
    key = tuple(labels.tolist())
    duplicate_of = seen.get(key, "")
    if not duplicate_of:
        seen[key] = name

    path = OUTPUT_DIR / f"{name}.csv"
    pd.DataFrame({"id": eval_df["id"], "answer": classes[labels]}).to_csv(path, index=False)
    counts = np.bincount(labels, minlength=len(classes))
    manifest.append(
        {
            "priority": priority,
            "name": name,
            "file": str(path),
            "desc": desc,
            "changes_vs_base": int((labels != base_labels).sum()),
            "count_min": int(counts.min()),
            "count_max": int(counts.max()),
            "duplicate_of": duplicate_of,
        }
    )
    print(priority, name, "changes", int((labels != base_labels).sum()), "dup", duplicate_of or "-")

# %%
# Candidatos v2 que faltaban por probar y son mas diversos.
for alpha, priority in [(0.38, 20), (0.42, 10), (0.46, 30), (0.50, 45)]:
    p = prior_correct(probs, train_prior, alpha)
    save(p.argmax(axis=1), f"submission_v3_prior_train_a{alpha:.2f}", f"prior train alpha={alpha:.2f}", priority)

for alpha, priority in [(0.38, 15), (0.42, 35), (0.46, 50)]:
    p = prior_correct(probs, uniform_prior, alpha)
    save(p.argmax(axis=1), f"submission_v3_prior_uniform_a{alpha:.2f}", f"prior uniform alpha={alpha:.2f}", priority)

# %%
# Temperatura y smooth mas diversos.
for temperature, alpha, priority in [
    (0.85, 0.35, 25),
    (0.85, 0.42, 55),
    (1.15, 0.42, 60),
    (1.30, 0.42, 70),
]:
    p = prior_correct(temperature_scale(probs, temperature), train_prior, alpha)
    save(
        p.argmax(axis=1),
        f"submission_v3_temp{temperature:.2f}_train_a{alpha:.2f}",
        f"temperature={temperature:.2f}, prior train alpha={alpha:.2f}",
        priority,
    )

for strength, alpha, priority in [
    (0.10, 0.30, 40),
    (0.12, 0.35, 65),
    (0.15, 0.35, 75),
]:
    p = prior_correct(neighbor_smooth(probs, strength), train_prior, alpha)
    save(
        p.argmax(axis=1),
        f"submission_v3_smooth{strength:.2f}_train_a{alpha:.2f}",
        f"neighbor smooth={strength:.2f}, prior train alpha={alpha:.2f}",
        priority,
    )

# %%
# Cuotas uniform/train. Uniform fue la familia de cuota que empato arriba.
for target_name, target_prior, settings in [
    ("uniform", uniform_prior, [(0.30, 5), (0.35, 1), (0.40, 32), (0.45, 80)]),
    ("train", train_prior, [(0.35, 12), (0.40, 42)]),
]:
    target_full = rounded_counts(target_prior, len(eval_df))
    for alpha, priority in settings:
        blended = (1 - alpha) * base_counts + alpha * target_full
        target = rounded_counts(blended / blended.sum(), len(eval_df))
        labels = rebalance_to_target(probs, target)
        save(
            labels,
            f"submission_v3_quota_{target_name}_a{alpha:.2f}",
            f"quota {target_name} blend alpha={alpha:.2f}",
            priority,
        )

# %%
manifest_df = pd.DataFrame(manifest).sort_values(["priority", "name"])
manifest_path = OUTPUT_DIR / "manifest_postproceso_beto384_v3.csv"
manifest_df.to_csv(manifest_path, index=False)

recommended = manifest_df[manifest_df["duplicate_of"].eq("")].head(10)
recommended_path = OUTPUT_DIR / "next_submit_postproceso_beto384_v3.csv"
recommended.to_csv(recommended_path, index=False)

print("manifest:", manifest_path)
print("next submit:", recommended_path)
print(recommended[["priority", "name", "changes_vs_base", "count_min", "count_max", "desc"]].to_string(index=False))
