"""
train_model.py — ScleraAI v4 Trainer
Uses in-memory score cache from main.py to avoid re-reading all images on every retrain.
Falls back to reading files if cache is empty (e.g. fresh server start).
"""
import json
import math
from datetime import datetime
from pathlib import Path

BASE_DIR    = "my_jaundice_dataset"
OUTPUT_FILE = "jaundice_ai_brain.json"
MIN_SAMPLES = 3


def compute_yellowness(r: float, g: float, b: float) -> float:
    return max(0.0, ((r + g) / (2.0 * max(b, 1.0)) - 1.0) * 10.0)


def extract_score(path: str) -> float | None:
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        w, h = img.size
        cx, cy = w // 2, h // 2
        region = img.crop((cx - 10, cy - 10, cx + 10, cy + 10))
        pixels = list(region.getdata())
        if not pixels: return None
        n = len(pixels)
        return compute_yellowness(
            sum(p[0] for p in pixels)/n,
            sum(p[1] for p in pixels)/n,
            sum(p[2] for p in pixels)/n
        )
    except Exception as exc:
        print(f"   ⚠ {path}: {exc}")
        return None


def collect_from_disk(category: str) -> list[float]:
    folder = Path(BASE_DIR) / category
    if not folder.exists(): return []
    return [s for f in folder.iterdir()
            if f.suffix.lower() in ('.jpg','.jpeg','.png')
            if (s := extract_score(str(f))) is not None]


def _mean(v): return sum(v) / len(v)
def _std(v, mu): return math.sqrt(sum((x-mu)**2 for x in v) / max(len(v)-1, 1))


def _accuracy_metrics(n_scores, j_scores, boundary) -> dict:
    tp = sum(1 for s in j_scores if s >  boundary)
    tn = sum(1 for s in n_scores if s <= boundary)
    fp = sum(1 for s in n_scores if s >  boundary)
    fn = sum(1 for s in j_scores if s <= boundary)
    total     = tp + tn + fp + fn
    accuracy  = (tp+tn)/total         if total    else 0
    precision = tp/(tp+fp)            if (tp+fp)  else 0
    recall    = tp/(tp+fn)            if (tp+fn)  else 0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0
    return {
        "train_accuracy": round(accuracy*100,  2),
        "precision":      round(precision*100, 2),
        "recall":         round(recall*100,    2),
        "f1":             round(f1*100,        2),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "warning": "TRAINING SET ONLY — not validated on held-out data"
    }


def run_dataset_training(score_cache: dict | None = None) -> dict | None:
    """
    score_cache: {"Normal": [floats], "Jaundice": [floats]}
    If cache is empty for a class, falls back to reading image files from disk.
    This prevents re-reading 200 images every single retrain.
    """
    # Use cache if populated, otherwise read disk
    n_scores = (score_cache or {}).get("Normal",   []) or collect_from_disk("Normal")
    j_scores = (score_cache or {}).get("Jaundice", []) or collect_from_disk("Jaundice")

    if len(n_scores) < MIN_SAMPLES or len(j_scores) < MIN_SAMPLES:
        print(f"⏳ Deferred — Normal:{len(n_scores)} Jaundice:{len(j_scores)} (need ≥{MIN_SAMPLES} each)")
        return None

    mu_n,  mu_j  = _mean(n_scores), _mean(j_scores)
    std_n, std_j = _std(n_scores, mu_n), _std(j_scores, mu_j)
    n_n,   n_j   = len(n_scores), len(j_scores)

    # Weighted mid-point boundary
    boundary = (mu_n * n_j + mu_j * n_n) / (n_n + n_j)
    metrics  = _accuracy_metrics(n_scores, j_scores, boundary)

    result = {
        "algorithm":                 "weighted_linear_discriminant",
        "optimal_decision_boundary": round(boundary, 4),
        "normal_mean":    round(mu_n,  4),
        "normal_std":     round(std_n, 4),
        "jaundice_mean":  round(mu_j,  4),
        "jaundice_std":   round(std_j, 4),
        "normal_count":   n_n,
        "jaundice_count": n_j,
        "normal_scores":  [round(s,2) for s in n_scores],
        "jaundice_scores":[round(s,2) for s in j_scores],
        "overlap_warning": mu_n >= mu_j,
        "metrics":         metrics,
        "trained_at":      datetime.utcnow().isoformat(),
        "status":          "DEPLOYED_AND_ACTIVE",
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=4)

    print(f"✅ Retrained | boundary={boundary:.3f} | acc={metrics['train_accuracy']}% | N={n_n} J={n_j}")
    return result


if __name__ == "__main__":
    run_dataset_training()
