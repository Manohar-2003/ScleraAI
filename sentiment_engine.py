"""
sentiment_engine.py — Facial Expression Analyser
Fixes:
  - mouth_width was computing sqrt(dx² + 0²) because p61[1]-p61[1] = 0 always
  - thresholds are now calibrated against face_scale so they're resolution-independent
"""

import math


def _pt(landmarks: list, idx: int, img_w: float, img_h: float) -> tuple[float, float]:
    pt = landmarks[idx]
    if isinstance(pt, dict):
        return pt.get("x", 0) * img_w, pt.get("y", 0) * img_h
    return getattr(pt, "x", 0) * img_w, getattr(pt, "y", 0) * img_h


def _dist(a: tuple, b: tuple) -> float:
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def analyze_facial_expression(
    landmarks: list,
    img_width: float,
    img_height: float,
) -> str:
    """
    Returns one of: "Surprised 😲", "Focused/Angry 😠", "Smiling/Happy 😊", "Neutral 😐"
    Falls back to Neutral on any error.
    """
    try:
        def pt(idx):
            return _pt(landmarks, idx, img_width, img_height)

        # --- Face scale: cheekbone width (landmark 234 → 454) ---
        face_scale = _dist(pt(234), pt(454)) or 1.0

        # --- Eye openness (vertical span, normalised) ---
        right_eye = _dist(pt(159), pt(145)) / face_scale   # upper-lid to lower-lid
        left_eye  = _dist(pt(386), pt(374)) / face_scale
        avg_eye   = (right_eye + left_eye) / 2.0

        # --- Eyebrow span (inter-brow distance, normalised) ---
        eyebrow_dist = _dist(pt(55), pt(285)) / face_scale

        # --- Mouth ratio: height / width (FIXED) ---
        # Mouth corners: 61 (right), 291 (left)
        # Mouth top: 0, bottom: 17
        mouth_width  = _dist(pt(61), pt(291)) or 1.0        # FIX: was using p61[1]-p61[1]=0
        mouth_height = _dist(pt(0),  pt(17))
        mouth_ratio  = mouth_height / mouth_width

        # --- Decision tree (thresholds normalised against face_scale) ---
        if avg_eye > 0.045:                                  # wide-open eyes → surprise
            return "Surprised 😲"
        elif eyebrow_dist < 0.20 or avg_eye < 0.024:        # furrowed / squinted → anger
            return "Focused/Angry 😠"
        elif mouth_ratio > 0.15:                             # open mouth height → smile
            return "Smiling/Happy 😊"
        else:
            return "Neutral 😐"

    except Exception:
        return "Neutral 😐"
