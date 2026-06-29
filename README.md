# ScleraAI
Open-source AI research tool detecting jaundice through webcam scleral (eye white) colour analysis. Actively collecting data — contribute your eye scan to help train the model.

# ScleraAI — Open Jaundice Detection Research

![Status](https://img.shields.io/badge/status-active%20data%20collection-yellow)
![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal)

> ⚗️ Research prototype. Not a medical device. Not clinically validated.

Jaundice — yellowing of the eye whites (sclera) — is one of the earliest 
visible signs of liver disease, hepatitis, and blood disorders. 
It is detectable visually before most patients realise they are sick.

ScleraAI is an open research project building a free, webcam-based 
jaundice screening dataset using scleral RGB colour analysis and 
MediaPipe face mesh tracking.

**We are in active data collection phase. The model is not yet validated.**
Every contribution brings us closer to a tool that could save lives — 
especially in regions where doctors and diagnostic labs are scarce.

---

## 🌍 We need your help — especially if you have jaundice

If you or someone you know has a confirmed jaundice diagnosis:
- Open the tool
- Go to **"Contribute your data"**
- Upload a clear photo of your eye whites
- Takes 30 seconds. Completely anonymous.

Your single image could be the one that makes this model work.

---

## How it works
Webcam / Uploaded image
->
MediaPipe FaceMesh (468 landmarks)
->
Scleral region isolation (nasal sclera, landmark 133→33)
->
Adaptive RGB sampling (scene-normalised brightness check)
->
Yellowness index = (R+G) / (2×B)
->
Weighted linear discriminant boundary
->
Classification + Gaussian likelihood ratio (when model is trained)
->
Background retrain after every contribution

---

## Two modes

| Mode | What it does |
|------|-------------|
| 🔍 **Predict** | Scans your eyes live and returns a result. Only active after ≥50 images per class. |
| 📊 **Collect** | Saves your eye crop to the training dataset. Works immediately. |

---

## Run locally in 3 steps

```bash
git clone https://github.com/YOUR_USERNAME/ScleraAI.git
cd ScleraAI
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`

> ⚠️ Always run via `uvicorn`. Do not open `index.html` directly 
> or via VS Code Live Server — the sentiment API will be offline.

---

## Tech stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + Python |
| Face tracking | MediaPipe FaceMesh |
| Charts | Chart.js |
| Classifier | Weighted linear discriminant + Gaussian likelihood ratio |
| Frontend | Vanilla JS — no framework |
| Data log | Append-only JSONL — safe to commit |

---

## Data collection status

| Class | Training images | Pending review |
|-------|----------------|----------------|
| Normal sclera | 0 | 0 |
| Jaundice sclera | 0 | 0 |

*Model activates at ≥50 verified images per class.*
*Update this table manually as you collect data.*

---

## Known limitations — full transparency

- Single RGB ratio formula, not deep learning
- Training-set accuracy only — no held-out validation yet  
- Lighting conditions affect readings
- Glasses can occlude the scleral sampling region
- Not validated on clinical data
- Requires ≥50 images per class before predictions display

---

## Privacy

- Face images never leave your device (webcam mode)
- Uploaded images stored locally, never committed to this repo
- Only anonymous score logs (`contributions.jsonl`) are tracked in git
- No accounts, no cookies, no tracking

---

## Contributing

**Have jaundice?** Upload your eye image in the tool.

**Healthy?** Use Collect → Normal mode to contribute normal sclera data.

**Doctor or researcher?** Open an issue. We can set up a controlled 
collection protocol with proper consent handling.

**Developer?** PRs welcome — especially for:
- Better scleral region polygon sampling
- Illumination normalisation
- CNN-based classifier (once dataset is large enough)

---

## Disclaimer

This tool is a research prototype. Results have no clinical validity.
Do not use as a substitute for medical advice or diagnosis.
If you suspect jaundice, see a doctor immediately.

---

## License

MIT — use freely, build on it, contribute back.
