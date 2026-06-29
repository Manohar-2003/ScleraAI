# ScleraAI — Open Jaundice Detection Research

> ⚗️ **Research prototype. Not a medical device. Not clinically validated.**

An open-source project building a webcam-based jaundice screening dataset using scleral (eye white) colour analysis.

---

## 🌍 We need your help

**Especially if you have jaundice.**

Every confirmed jaundice eye image contributes to training a model that could help detect jaundice earlier — particularly in regions where doctors are scarce.

👉 **[Open the tool](http://127.0.0.1:8000) → Click "Contribute your data"**

---

## How it works

```
Webcam → MediaPipe FaceMesh → Scleral RGB sampling
→ Yellowness index → Classification against trained boundary
→ Image saved to dataset → Background retrain → Boundary updates
```

**Two modes:**
- **Predict** — scan your eyes and get a result (requires ≥50 images per class)
- **Collect** — contribute your eye images to train the model

---

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# Open http://127.0.0.1:8000
```

> ⚠️ Always run via `uvicorn`, not VS Code Live Server. The frontend fetches from port 8000.

---

## Architecture

```
main.py          — FastAPI server. /predict, /save-photo, /upload-contribution, /analytics
train_model.py   — Weighted linear discriminant trainer. Runs in background thread after every save.
sentiment_engine.py — Facial expression detection (MediaPipe landmarks)
index.html       — Single-page frontend. No framework. MediaPipe + Chart.js.
contributions.jsonl — Anonymised score log (append-only). Safe to commit.
upload_log.jsonl    — Upload metadata (no images). Safe to commit.
my_jaundice_dataset/ — Face images. NEVER committed (see .gitignore)
```

---

## Data collection status

| Class    | Training images | Pending review |
|----------|----------------|----------------|
| Normal   | 0              | 0              |
| Jaundice | 0              | 0              |

*Model activates at ≥50 images per class.*

---

## Known limitations (be honest with your audience)

- Uses a single RGB ratio formula — no deep learning yet
- Not validated on held-out data
- Lighting conditions affect readings significantly
- Glasses can block the scleral sampling region
- Training-set accuracy only — not a real accuracy metric
- Requires ≥50 images per class before predictions are shown

---

## Contributing data

**If you have jaundice:**
1. Open the tool
2. Go to "Contribute your data"
3. Upload a clear photo of your eye whites
4. Check "Doctor confirmed" if applicable
5. Submit — your contribution goes into pending review

**If you're healthy:**
- Use Collect → Normal mode in the live screener

**If you're a doctor or researcher:**
- Open an issue to discuss dataset collaboration
- We can set up a controlled collection protocol

---

## Privacy

- Face images are stored locally only and never committed to this repo
- Only anonymous score logs (`contributions.jsonl`) are tracked in git
- Upload images go into `my_jaundice_dataset/pending_review/` — local only
- No user accounts, no tracking, no cookies

---

## Disclaimer

This tool is a research prototype. Results have no clinical validity. Do not use this as a substitute for medical advice, diagnosis, or treatment. If you suspect jaundice, see a doctor immediately.

---

## License

MIT — use freely, contribute openly, attribute honestly.
