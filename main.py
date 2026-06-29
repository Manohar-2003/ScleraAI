"""
main.py — ScleraAI v5 — Public Launch Build
Run: uvicorn main:app --reload
Open: http://127.0.0.1:8000
"""

import base64
import json
import math
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import sentiment_engine
from train_model import run_dataset_training

# ── Config ──────────────────────────────────────────────────────────
BRAIN_FILE       = "jaundice_ai_brain.json"
CONTRIBUTIONS    = "contributions.jsonl"   # append-only, one line per entry
DATASET_DIR      = "my_jaundice_dataset"
DEFAULT_BOUNDARY = 4.5
MIN_SAMPLES_PREDICT = 50   # Never show predictions below this threshold

app = FastAPI(title="ScleraAI", version="5.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── Thread safety ────────────────────────────────────────────────────
_lock = threading.Lock()

# ── State ────────────────────────────────────────────────────────────
_save_counter    = 0
_retrain_running = False
_last_retrain    = 0.0
_history: list[dict] = []
_score_cache: dict[str, list[float]] = {"Normal": [], "Jaundice": []}


def _load_brain() -> dict:
    try:
        with open(BRAIN_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


brain             = _load_brain()
decision_boundary = float(brain.get("optimal_decision_boundary", DEFAULT_BOUNDARY))
normal_mean       = float(brain.get("normal_mean",   0.0))
jaundice_mean     = float(brain.get("jaundice_mean", 0.0))
normal_std        = float(brain.get("normal_std",    1.0))
jaundice_std      = float(brain.get("jaundice_std",  1.0))
normal_count      = int(brain.get("normal_count",    0))
jaundice_count    = int(brain.get("jaundice_count",  0))

if brain.get("optimal_decision_boundary"):
    _history.append({
        "boundary":      brain["optimal_decision_boundary"],
        "normal_mean":   brain.get("normal_mean", 0),
        "jaundice_mean": brain.get("jaundice_mean", 0),
        "ts":            brain.get("trained_at", "startup"),
    })
    with _lock:
        _score_cache["Normal"]   = brain.get("normal_scores",   [])
        _score_cache["Jaundice"] = brain.get("jaundice_scores", [])


# ── Contributions (append-only JSONL — never rewrites whole file) ────
def _append_contribution(score: float, label: str, source: str = "webcam"):
    entry = json.dumps({
        "score":  round(score, 4),
        "label":  label,
        "source": source,
        "ts":     datetime.utcnow().isoformat(),
    })
    with open(CONTRIBUTIONS, "a") as f:
        f.write(entry + "\n")


def _count_contributions() -> int:
    try:
        with open(CONTRIBUTIONS) as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return 0


# ── Background retrain ───────────────────────────────────────────────
def _background_retrain():
    global decision_boundary, brain, _retrain_running, _last_retrain
    global _history, normal_mean, jaundice_mean, normal_std, jaundice_std
    global normal_count, jaundice_count
    _retrain_running = True
    try:
        with _lock:
            cache_copy = {
                "Normal":   list(_score_cache["Normal"]),
                "Jaundice": list(_score_cache["Jaundice"]),
            }
        result = run_dataset_training(cache_copy)
        if result:
            decision_boundary = result["optimal_decision_boundary"]
            normal_mean       = result.get("normal_mean",   0.0)
            jaundice_mean     = result.get("jaundice_mean", 0.0)
            normal_std        = result.get("normal_std",    1.0)
            jaundice_std      = result.get("jaundice_std",  1.0)
            normal_count      = result.get("normal_count",  0)
            jaundice_count    = result.get("jaundice_count",0)
            brain             = _load_brain()
            _last_retrain     = time.time()
            _history.append({
                "boundary":      result["optimal_decision_boundary"],
                "normal_mean":   result.get("normal_mean", 0),
                "jaundice_mean": result.get("jaundice_mean", 0),
                "ts":            datetime.utcnow().isoformat(),
            })
            if len(_history) > 50:
                _history.pop(0)
    finally:
        _retrain_running = False


def _trigger_retrain():
    if not _retrain_running:
        threading.Thread(target=_background_retrain, daemon=True).start()


# ── Classifier ───────────────────────────────────────────────────────
def _classify(score: float) -> dict:
    total_samples = normal_count + jaundice_count
    model_ready   = (normal_count >= MIN_SAMPLES_PREDICT and
                     jaundice_count >= MIN_SAMPLES_PREDICT)

    if not model_ready:
        needed = MIN_SAMPLES_PREDICT
        return {
            "result":     "INSUFFICIENT_DATA",
            "label":      "Not enough data",
            "confidence": 0,
            "score":      round(score, 3),
            "boundary":   round(decision_boundary, 3),
            "distance":   0,
            "normal_probability":   0,
            "jaundice_probability": 0,
            "model":      "none",
            "warning":    f"Model needs ≥{needed} images per class to predict. "
                          f"Currently: Normal={normal_count}, Jaundice={jaundice_count}.",
        }

    is_jaundice = score > decision_boundary
    distance    = abs(score - decision_boundary)

    if normal_mean > 0 and jaundice_mean > 0:
        def gaussian(x, mu, sigma):
            sigma = max(sigma, 0.01)
            return math.exp(-0.5*((x-mu)/sigma)**2) / (sigma*math.sqrt(2*math.pi))

        p_n = gaussian(score, normal_mean,   normal_std)
        p_j = gaussian(score, jaundice_mean, jaundice_std)
        total = p_n + p_j + 1e-9

        normal_pct   = round(p_n/total*100, 1)
        jaundice_pct = round(p_j/total*100, 1)

        # Gaussian overrides boundary decision
        is_jaundice  = jaundice_pct > normal_pct
        confidence   = jaundice_pct if is_jaundice else normal_pct

        explanation = {
            "normal_probability":   normal_pct,
            "jaundice_probability": jaundice_pct,
            "normal_mean":          round(normal_mean,   2),
            "jaundice_mean":        round(jaundice_mean, 2),
            "score_vs_normal":      round(score - normal_mean,   2),
            "score_vs_jaundice":    round(score - jaundice_mean, 2),
            "model": "gaussian_likelihood_ratio",
        }
    else:
        raw_conf = min(99.0, 50.0 + distance * 14.0)
        explanation = {
            "normal_probability":   round(100-raw_conf,1) if is_jaundice else round(raw_conf,1),
            "jaundice_probability": round(raw_conf,1)     if is_jaundice else round(100-raw_conf,1),
            "model":   "boundary_distance_fallback",
            "warning": "Using fallback classifier. Collect more data.",
        }
        confidence = raw_conf

    return {
        "result":     "JAUNDICE" if is_jaundice else "NORMAL",
        "label":      "Elevated Risk" if is_jaundice else "Normal Sclera",
        "confidence": round(confidence, 1),
        "score":      round(score, 3),
        "boundary":   round(decision_boundary, 3),
        "distance":   round(distance, 3),
        **explanation,
    }


# ── Schemas ──────────────────────────────────────────────────────────
class SavePhotoPackage(BaseModel):
    image_data: str
    score: float
    label: str
    source: str = "webcam"      # "webcam" | "upload"
    doctor_confirmed: bool = False

class UploadContribution(BaseModel):
    image_data: str             # base64 data-URL
    label: str                  # "Jaundice" | "Normal"
    doctor_confirmed: bool
    notes: str = ""

class PredictPackage(BaseModel):
    score: float

class SentimentPackage(BaseModel):
    landmarks: list


# ── Routes ───────────────────────────────────────────────────────────
@app.get("/")
async def get_index():
    if not Path("index.html").exists():
        raise HTTPException(404, "Run from project folder: uvicorn main:app --reload")
    return FileResponse("index.html")


@app.get("/boundary")
async def get_boundary():
    return {
        "boundary":        decision_boundary,
        "retrain_running": _retrain_running,
        "has_model":       normal_count >= MIN_SAMPLES_PREDICT and jaundice_count >= MIN_SAMPLES_PREDICT,
        "normal_count":    normal_count,
        "jaundice_count":  jaundice_count,
        "needed":          MIN_SAMPLES_PREDICT,
    }


@app.post("/predict")
async def predict(payload: PredictPackage):
    return JSONResponse(_classify(payload.score))


@app.post("/save-photo")
async def save_photo(payload: SavePhotoPackage):
    global _save_counter
    if payload.label not in ("Normal", "Jaundice"):
        raise HTTPException(400, "label must be 'Normal' or 'Jaundice'")
    try:
        _, encoded = payload.image_data.split(",", 1)
        data = base64.b64decode(encoded)
    except Exception as exc:
        raise HTTPException(400, f"Bad image data: {exc}")

    save_dir = Path(DATASET_DIR) / payload.label
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / f"frame_{uuid.uuid4().hex}.jpg").write_bytes(data)

    with _lock:
        _score_cache[payload.label].append(round(payload.score, 4))

    _append_contribution(payload.score, payload.label, payload.source)
    _save_counter += 1
    _trigger_retrain()

    return JSONResponse({
        "status":          "saved",
        "label":           payload.label,
        "saves_total":     _save_counter,
        "retrain_running": _retrain_running,
    })


@app.post("/upload-contribution")
async def upload_contribution(payload: UploadContribution):
    """
    Accepts image file uploads from people with jaundice.
    Stored in a separate 'uploaded' subfolder for manual review
    before being merged into training data.
    """
    if payload.label not in ("Normal", "Jaundice"):
        raise HTTPException(400, "label must be 'Normal' or 'Jaundice'")
    try:
        _, encoded = payload.image_data.split(",", 1)
        data = base64.b64decode(encoded)
    except Exception as exc:
        raise HTTPException(400, f"Bad image data: {exc}")

    # Save to pending review folder — NOT auto-merged into training
    review_dir = Path(DATASET_DIR) / "pending_review" / payload.label
    review_dir.mkdir(parents=True, exist_ok=True)
    fname = f"upload_{uuid.uuid4().hex}.jpg"
    (review_dir / fname).write_bytes(data)

    # Log the contribution metadata
    entry = json.dumps({
        "file":             fname,
        "label":            payload.label,
        "doctor_confirmed": payload.doctor_confirmed,
        "notes":            payload.notes[:500],   # cap notes length
        "source":           "upload",
        "ts":               datetime.utcnow().isoformat(),
    })
    with open("upload_log.jsonl", "a") as f:
        f.write(entry + "\n")

    return JSONResponse({
        "status":  "received",
        "message": "Thank you. Your contribution is pending review before being added to training data.",
        "label":   payload.label,
        "doctor_confirmed": payload.doctor_confirmed,
    })


@app.get("/analytics")
async def get_analytics():
    def count_images(cat: str) -> int:
        d = Path(DATASET_DIR) / cat
        return sum(1 for f in d.iterdir()
                   if f.suffix.lower() in ('.jpg','.jpeg','.png')) if d.exists() else 0

    def count_pending(cat: str) -> int:
        d = Path(DATASET_DIR) / "pending_review" / cat
        return sum(1 for f in d.iterdir()
                   if f.suffix.lower() in ('.jpg','.jpeg','.png')) if d.exists() else 0

    return JSONResponse({
        "normal_count":          count_images("Normal"),
        "jaundice_count":        count_images("Jaundice"),
        "pending_normal":        count_pending("Normal"),
        "pending_jaundice":      count_pending("Jaundice"),
        "total_contribs":        _count_contributions(),
        "boundary":              decision_boundary,
        "retrain_running":       _retrain_running,
        "last_retrain":          _last_retrain,
        "normal_mean":           brain.get("normal_mean"),
        "normal_std":            brain.get("normal_std"),
        "jaundice_mean":         brain.get("jaundice_mean"),
        "jaundice_std":          brain.get("jaundice_std"),
        "overlap_warning":       brain.get("overlap_warning", False),
        "normal_scores":         brain.get("normal_scores",   []),
        "jaundice_scores":       brain.get("jaundice_scores", []),
        "history":               _history,
        "metrics":               brain.get("metrics"),
        "has_model":             normal_count >= MIN_SAMPLES_PREDICT and jaundice_count >= MIN_SAMPLES_PREDICT,
        "min_samples_needed":    MIN_SAMPLES_PREDICT,
    })


@app.post("/retrain")
async def trigger_retrain():
    if _retrain_running:
        return JSONResponse({"status": "already_running", "boundary": decision_boundary})
    _trigger_retrain()
    return JSONResponse({"status": "started", "boundary": decision_boundary})


@app.post("/get-sentiment")
async def get_sentiment(payload: SentimentPackage):
    try:
        emotion = "Neutral 😐"
        if payload.landmarks:
            emotion = sentiment_engine.analyze_facial_expression(payload.landmarks, 1280, 720)
        return {"status": "success", "server_sentiment": emotion}
    except Exception as exc:
        return {"status": "error", "server_sentiment": "Neutral 😐", "detail": str(exc)}
