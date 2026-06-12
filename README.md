#  ForensicAI — Automated Crime Scene Reconstruction & Event Inference

> An AI-powered forensic analysis system that detects evidence from crime scene images using computer vision and generates structured event reconstructions using large language models.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=flat&logo=ultralytics&logoColor=black)
![Groq](https://img.shields.io/badge/Groq-LLaMA4-orange?style=flat)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)

---

##  What It Does

Upload a crime scene image and the system will:

1. **Detect evidence** — bloodstains, weapons, persons, shell casings, bags, furniture
2. **Enhance the image** — CLAHE preprocessing, heatmap overlay, evidence grid
3. **Reconstruct the event** — LLaMA-4 Scout (via Groq) analyzes detections and generates a structured forensic narrative including probable sequence of events, key evidence, and investigative leads

---

##  Architecture

```
Image Input
    │
    ▼
┌─────────────────────────────────────┐
│         Detection Pipeline          │
│  1. blood_best.pt   (custom)        │
│  2. weapon_best.pt  (custom)        │
│  3. yolov8m.pt      (general)       │
│  4. Roboflow API    (optional)      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│       Groq — LLaMA-4 Scout          │
│   Vision + Forensic Reasoning       │
└─────────────────────────────────────┘
    │
    ▼
  Forensic Report + Event Timeline
```

---

##  Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | YOLOv8m (Ultralytics) + Custom `.pt` models |
| AI Forensic Analysis | Groq API — `llama-4-scout-17b-16e-instruct` |
| Image Preprocessing | OpenCV, CLAHE enhancement |
| Backend | Flask (Python) |
| Optional Enhancement | Roboflow API |

---

##  Detection Stack

| Priority | Model | Detects |
|----------|-------|---------|
| 1 | `blood_best.pt` *(custom trained)* | Bloodstains, biological evidence |
| 2 | `weapon_best.pt` *(custom trained)* | Guns, knives, weapons |
| 3 | `yolov8m.pt` *(YOLO medium)* | Persons, furniture, bags, vehicles |
| 4 | Roboflow API *(optional)* | Shell casings, enhanced weapon detection |

---

##  Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/PPiyush14/ForensicAI---Automated-Crime-Scene-Reconstruction-and-Event-Inference-Using-Computer-Vision.git
cd ForensicAI---Automated-Crime-Scene-Reconstruction-and-Event-Inference-Using-Computer-Vision
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up API keys
Create a `.env` file in the root directory:
```env
# Required — get free key at https://console.groq.com (no credit card)
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx

# Optional — enhanced detection via Roboflow
ROBOFLOW_API_KEY=your_roboflow_key
```

### 4. Add model files
Place your custom trained models in the root directory:
```
blood_best.pt
weapon_best.pt
```
> `yolov8m.pt` will auto-download on first run via Ultralytics.

### 5. Run
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

---

##  Project Structure

```
├── app.py                  # Flask app entry point
├── config.py               # Configuration and model paths
├── routes.py               # API routes
├── requirements.txt
├── .env                    # API keys (not committed)
├── models/                 # Custom .pt model files
├── utils/
│   ├── preprocessor.py     # CLAHE, heatmap, evidence grid
│   └── __init__.py
├── templates/
│   └── index.html          # Web UI
└── static/                 # Assets
```

---

## 🤖 Why Groq + LLaMA-4 Scout?

| Service | Model | Vision | Free Tier |
|---------|-------|--------|-----------|
| **Groq ✅** | LLaMA-4 Scout | ✅ | 30 req/min, no daily cap |
| Gemini | Flash 2.0 | ✅ | Hits daily quota quickly |
| OpenAI | GPT-4o | ✅ | Paid only |

Groq offers the fastest inference with full vision support and no hard daily limits on the free tier — ideal for local development and demos.

---

## 👤 Author

**Piyush Rajvaidya**  
[![GitHub](https://img.shields.io/badge/GitHub-PPiyush14-black?logo=github)](https://github.com/PPiyush14)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Piyush%20Rajvaidya-blue?logo=linkedin)](https://www.linkedin.com/in/piyush-rajvaidya/)
