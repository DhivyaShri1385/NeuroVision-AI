<div align="center">

<img src="https://img.shields.io/badge/NeuroVision_AI-v1.0.0-58a6ff?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZmZmIiBzdHJva2Utd2lkdGg9IjIiPjxwYXRoIGQ9Ik05LjUgMkM2LjUgMiA0IDQuNSA0IDcuNWMwIDEuNS41IDMgMS41IDRBNS41IDUuNSAwIDAgMSAzIDEwYzAgMS43IDEgMy4yIDIuNSA0LS4zLjctLjUgMS40LS41IDIgMCAzIDIuNSA1LjUgNS41IDUuNUgxNGMzIDAgNS41LTIuNSA1LjUtNS41IDAtLjYtLjItMS4zLS41LTJDMTQ4IDEzLjIgMTU4IDExLjcgMTU4IDEwYTUuNSA1LjUgMCAwIDEtMi41LTAuNSIgLz48L3N2Zz4=" alt="NeuroVision AI" />

# NeuroVision AI

### Explainable Brain Tumour Diagnosis & Segmentation Platform

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Phases](https://img.shields.io/badge/All_7_Phases-Complete-3fb950?style=flat-square)](#development-phases)

<br/>

**[🚀 Live Demo](https://dhivyashri1385.github.io/NeuroVision-AI/) · [📖 API Docs](http://localhost:8000/docs) · [📊 Report Bug](https://github.com/DhivyaShri1385/NeuroVision-AI/issues)**

<br/>

> A production-quality AI healthcare platform that classifies brain MRI scans, segments tumour regions, and explains every prediction — all served through a REST API and a sleek dark-theme dashboard.

</div>

---

## ✨ What It Does

| Capability | Description |
|---|---|
| 🧠 **Classify** | 4-class MRI classification (Glioma · Meningioma · No Tumour · Pituitary) |
| 🎯 **Segment** | Pixel-level tumour masking with Attention U-Net (31.6M params) |
| 🔍 **Explain** | Grad-CAM · SmoothGrad · Integrated Gradients heatmaps |
| 📄 **Report** | One-click clinical PDF report (diagnosis + overlays + metrics) |
| ⚡ **Serve** | FastAPI REST backend — `/classify` · `/segment` · `/explain` · `/report` |
| 🎨 **Visualise** | React glassmorphism dashboard with real-time inference |
| 🐳 **Deploy** | Docker + docker-compose · GitHub Pages frontend |

---

## 🖥️ Dashboard Preview

<div align="center">

```
╔══════════════════════════════════════════════════════════════════╗
║  🧠 NeuroVision AI                          v1.0.0  ⬡ GitHub   ║
╠══════════════════════════════════════════════════════════════════╣
║  ● API connected  [classifier(EfficientNetB3)]  [GradCAM]       ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌─────────────────────┐  ┌────────────────────────────────────┐ ║
║  │  📁 MRI Image       │  │  ⏱ 487ms  482ms  —  1269ms        │ ║
║  │                     │  ├──────────────────────────────────  │ ║
║  │   ┌─────────────┐   │  │  🧠 Classify  🔬 Segment  ⚡ XAI  │ ║
║  │   │  [MRI SCAN] │   │  │                                    │ ║
║  │   └─────────────┘   │  │  ╔══════════════════════════════╗  │ ║
║  │                     │  │  ║  GLIOMA              87.4%  ║  │ ║
║  │  ⚡ XAI Method      │  │  ╟──────────────────────────────╢  │ ║
║  │  ● Grad-CAM  fast   │  │  ║  glioma    ████████  87.4%  ║  │ ║
║  │  ○ SmoothGrad       │  │  ║  mening    ██        7.2%   ║  │ ║
║  │                     │  │  ║  no_tumor  █         3.1%   ║  │ ║
║  │  [  Analyse MRI  ]  │  │  ║  pituitary █         2.3%   ║  │ ║
║  └─────────────────────┘  │  ╚══════════════════════════════╝  │ ║
║                           └────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════╝
```

</div>

---

## 🏗️ Tech Stack

| Layer | Technology | Version |
|---|---|---|
| **Classification** | EfficientNetB3 + ResNet50 + DenseNet121 Ensemble | TF 2.20 |
| **Segmentation** | Attention U-Net (Oktay et al. 2018) | 31.6M params |
| **Explainability** | Grad-CAM · SmoothGrad · Integrated Gradients | Phase 4 |
| **Backend** | FastAPI + Pydantic v2 + Uvicorn | 0.104 |
| **Frontend** | React 19 + Vite 8 + Recharts | Glassmorphism UI |
| **Training** | TensorFlow 2.20 + Albumentations 2.0.8 | Python 3.12 |
| **PDF Reports** | ReportLab 4.x | Dark-theme clinical PDF |
| **Deployment** | Docker + Nginx + docker-compose | GitHub Pages |
| **Logging** | Loguru structured logging | Rotating 50 MB |

---

## 📦 Dataset

[Brain Tumour MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — **4 perfectly balanced classes, 1,932 images total.**

| Split | Glioma | Meningioma | No Tumour | Pituitary | **Total** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Train | 335 | 335 | 335 | 335 | **1,340** |
| Val   | 99  | 99  | 99  | 99  | **396**   |
| Test  | 49  | 49  | 49  | 49  | **196**   |
| **Total** | **483** | **483** | **483** | **483** | **1,932** |

---

## 🗂️ Project Structure

```
neurovision-ai/
├── 📁 configs/
│   └── config.yaml                   # Single source of truth for all hyperparams
│
├── 📁 src/
│   ├── 📁 data/
│   │   ├── dataset_organizer.py      # Audit & integrity check
│   │   ├── preprocessor.py           # Resize, normalise, grayscale→RGB
│   │   ├── augmentation.py           # Albumentations 2.0.8 medical pipeline
│   │   ├── splitter.py               # Split validation & CSV manifests
│   │   └── data_loader.py            # tf.data pipeline (AUTOTUNE, prefetch)
│   │
│   ├── 📁 models/                    # Phase 2 — EfficientNetB3, ResNet50, DenseNet121
│   │   ├── classifier.py             # Backbone builders + unfreeze logic
│   │   └── ensemble.py               # Soft-voting ensemble predictor
│   │
│   ├── 📁 segmentation/              # Phase 3 — Attention U-Net
│   │   ├── unet.py                   # Standard U-Net
│   │   ├── attention_unet.py         # Attention U-Net (31.6M params)
│   │   ├── losses.py                 # Dice · BCE+Dice · Focal · Tversky
│   │   ├── metrics.py                # DiceCoefficient · IoU · PixelAccuracy
│   │   ├── mask_generator.py         # Pseudo-mask via Otsu + morphology
│   │   ├── data_loader.py            # (image, mask) tf.data pipeline
│   │   ├── trainer.py                # Training loop + callbacks
│   │   └── visualizer.py             # History · overlays · metrics bar
│   │
│   ├── 📁 xai/                       # Phase 4 — Explainability
│   │   ├── gradcam.py                # Grad-CAM + Grad×Input fallback
│   │   ├── saliency.py               # VanillaSaliency · SmoothGrad · IG
│   │   └── visualizer.py             # Overlay · comparison · summary grid
│   │
│   ├── 📁 reports/                   # Phase 7 — PDF generation
│   │   └── generator.py              # ReportLab clinical PDF
│   │
│   ├── 📁 evaluation/
│   │   ├── metrics.py                # Accuracy · F1 · AUC · MCC · Kappa
│   │   └── visualizer.py             # Confusion matrix · ROC · per-class bars
│   │
│   └── 📁 utils/
│       ├── logger.py                 # Loguru structured logging
│       └── config.py                 # Typed frozen AppConfig dataclass
│
├── 📁 app/                           # Phase 5 — FastAPI backend
│   ├── main.py                       # App entrypoint · CORS · lifespan
│   ├── model_registry.py             # Singleton model loader
│   ├── schemas.py                    # Pydantic v2 request/response models
│   ├── preprocessing.py              # Image → tensor helpers
│   └── routers/
│       ├── classify.py               # POST /classify
│       ├── segment.py                # POST /segment
│       ├── explain.py                # POST /explain
│       ├── report.py                 # POST /report  → PDF download
│       └── health.py                 # GET  /health
│
├── 📁 frontend/                      # Phase 6 — React dashboard
│   ├── src/
│   │   ├── App.jsx                   # Main layout · demo mode · nav
│   │   ├── components/
│   │   │   ├── HeroSection.jsx       # Animated brain orb landing
│   │   │   ├── UploadPanel.jsx       # Drag-and-drop MRI uploader
│   │   │   ├── ClassificationCard.jsx # Gradient probability bars
│   │   │   ├── ImageViewer.jsx       # Lightbox · download
│   │   │   ├── AnalysisResult.jsx    # Tabbed results dashboard
│   │   │   └── StatusBar.jsx         # Live API health indicator
│   │   └── api/neurovision.js        # Axios API client
│   └── vite.config.js                # /NeuroVision-AI/ base for Pages
│
├── 📁 scripts/
│   ├── run_phase1.py                 # Dataset audit & EDA
│   ├── run_phase2.py                 # Classifier training
│   ├── run_phase3.py                 # Segmentation training
│   ├── run_phase4.py                 # XAI generation
│   ├── run_phase5.py                 # Start API server
│   ├── run_phase6.py                 # Start frontend dev server
│   └── run_phase7.py                 # Generate PDF reports
│
├── Dockerfile                        # Python 3.12-slim API image
├── docker-compose.yml                # api + frontend services
└── .github/workflows/
    └── deploy-frontend.yml           # Auto-deploy to GitHub Pages
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12, pip
- Node.js 20+, npm
- [Brain Tumour MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

### 1 · Clone & Install

```bash
git clone https://github.com/DhivyaShri1385/NeuroVision-AI.git
cd NeuroVision-AI/neurovision-ai
pip install -r requirements.txt
npm --prefix frontend install
```

### 2 · Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:
```env
DATASET_ROOT=C:/path/to/brain-tumor-dataset
PROJECT_ROOT=C:/path/to/neurovision-ai
```

### 3 · Run Training Pipeline

```bash
# Phase 1 — Preprocessing & EDA
python scripts/run_phase1.py

# Phase 2 — Train EfficientNetB3 classifier
python scripts/run_phase2.py --backbone EfficientNetB3

# Phase 3 — Train Attention U-Net segmentation
python scripts/run_phase3.py

# Phase 4 — Generate XAI explanations
python scripts/run_phase4.py --n-samples 20
```

### 4 · Run the Full Stack

```bash
# Terminal 1 — API server
python scripts/run_phase5.py
# → http://localhost:8000/docs

# Terminal 2 — React dashboard
npm --prefix frontend run dev
# → http://localhost:3000
```

### 5 · Docker (one command)

```bash
docker-compose up --build
# API  → http://localhost:8000/docs
# UI   → http://localhost:3000
```

### 6 · Generate PDF Reports

```bash
python scripts/run_phase7.py --n 5
# → outputs/reports/*.pdf
```

---

## 🧠 Model Architectures

### Classification — Two-Stage Transfer Learning

```
Input (224 × 224 × 3)
        │
        ▼
EfficientNetB3  ──  ImageNet pretrained
  Stage 1: backbone frozen, train head only  (lr = 1e-3, 10 epochs)
  Stage 2: unfreeze top 30 layers            (lr = 1e-5, 70 epochs)
        │
GlobalAveragePooling2D
        │
BatchNorm → Dense(512, ReLU) → Dropout(0.4)
        │
Dense(4, Softmax)
        │
   [Glioma | Meningioma | No Tumour | Pituitary]
```

**Ensemble:** EfficientNetB3 + ResNet50 + DenseNet121 → soft-vote (average probabilities)

### Segmentation — Attention U-Net

```
Input (256 × 256 × 1)   grayscale MRI
        │
  Encoder (filters: 64 → 128 → 256 → 512)
  MaxPool × 4
        │
  Bottleneck (1024 filters)
        │
  Decoder with Attention Gates ←── Skip connections
  (512 → 256 → 128 → 64)
        │
  Conv1×1 → Sigmoid
        │
  Binary Mask (256 × 256 × 1)   tumour / background
```

**Attention Gate:** `alpha = σ(W_θ(x) + W_φ(g))` · filters irrelevant background regions

---

## ⚙️ Training Hyperparameters

### Classifier

| Parameter | Value |
|---|---|
| Backbone | EfficientNetB3 (11.6M params) |
| Pretrained weights | ImageNet |
| Input size | 224 × 224 × 3 |
| Batch size | 32 |
| Epochs | 80 + early stopping (patience=12) |
| Stage 1 LR | 1e-3 (head only) |
| Stage 2 LR | 1e-5 (top 30 layers) |
| Loss | Categorical cross-entropy (label smoothing=0.1) |
| Augmentation | Rotation ±15° · H-flip · Elastic · Brightness · GaussNoise |

### Segmentation

| Parameter | Value |
|---|---|
| Architecture | Attention U-Net |
| Parameters | 31.6 M |
| Input size | 256 × 256 × 1 |
| Loss | BCE + Dice (combined) |
| Metrics | Dice coefficient · IoU · Pixel accuracy |
| Batch size | 16 |
| Epochs | 50 + early stopping (patience=10) |
| Learning rate | 1e-4 → reduce on plateau |

---

## 🔌 API Reference

Start the server: `python scripts/run_phase5.py`
Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

| Method | Endpoint | Description | Returns |
|---|---|---|---|
| `GET` | `/health` | Liveness probe | Models loaded |
| `GET` | `/classes` | Class names | `["glioma", ...]` |
| `POST` | `/classify` | Classify MRI image | Class + probabilities |
| `POST` | `/segment` | Segment tumour region | Mask + overlay (base64 PNG) |
| `POST` | `/explain` | Grad-CAM / SmoothGrad | Heatmap + overlay (base64 PNG) |
| `POST` | `/report` | Generate clinical PDF | `.pdf` download |
| `POST` | `/analyze` | Full pipeline (all above) | Combined JSON |

**Example — classify:**
```bash
curl -X POST http://localhost:8000/classify \
  -F "file=@brain_mri.jpg" | python -m json.tool
```

```json
{
  "predicted_class": "glioma",
  "confidence": 0.874,
  "probabilities": {
    "glioma": 0.874, "meningioma": 0.072,
    "no_tumor": 0.031, "pituitary": 0.023
  },
  "model_name": "EfficientNetB3",
  "inference_time_ms": 487.3
}
```

---

## 📊 Evaluation Metrics

### Classification
- **Accuracy** · Precision · Recall · **F1-Score** (per-class + macro)
- **ROC-AUC** (one-vs-rest, macro average)
- Confusion Matrix · Matthews Correlation Coefficient · Cohen's Kappa

### Segmentation
- **Dice Coefficient** (soft + hard threshold)
- **IoU / Jaccard Index**
- Pixel Accuracy · Precision · Recall

---

## 🔍 Explainability Methods

| Method | Algorithm | Speed |
|---|---|---|
| **Grad-CAM** | Gradient × Input (class-discriminative) | ~500 ms |
| **SmoothGrad** | Average gradient over 50 noisy copies (Smilkov et al. 2017) | ~25 s |
| **Integrated Gradients** | Trapezoidal integration: baseline→input (Sundararajan et al. 2017) | ~8 s |

---

## 📋 Development Phases

| # | Phase | Files | Status |
|---|---|---|---|
| **1** | Dataset · Preprocessing · Augmentation · EDA | `src/data/` · `scripts/run_phase1.py` | ✅ **Complete** |
| **2** | EfficientNetB3 · ResNet50 · DenseNet121 Ensemble | `src/models/` · `scripts/run_phase2.py` | ✅ **Complete** |
| **3** | Attention U-Net · Dice/IoU · Pseudo-masks | `src/segmentation/` · `scripts/run_phase3.py` | ✅ **Complete** |
| **4** | Grad-CAM · SmoothGrad · Integrated Gradients | `src/xai/` · `scripts/run_phase4.py` | ✅ **Complete** |
| **5** | FastAPI · `/classify` `/segment` `/explain` `/report` | `app/` · `scripts/run_phase5.py` | ✅ **Complete** |
| **6** | React Dashboard · Glassmorphism UI · Demo mode | `frontend/` · GitHub Pages | ✅ **Complete** |
| **7** | ReportLab PDF · Docker · docker-compose · CI/CD | `Dockerfile` · `.github/workflows/` | ✅ **Complete** |

---

## 🌐 Deployment

### GitHub Pages (Frontend)

The React dashboard auto-deploys to GitHub Pages on every push to `main`:

```
https://dhivyashri1385.github.io/NeuroVision-AI/
```

> **Note:** The live demo runs in **demo mode** (no backend). To get real predictions, run the backend locally with `python scripts/run_phase5.py`.

### Docker (Full Stack)

```bash
# Build and start both services
docker-compose up --build

# API  →  http://localhost:8000
# UI   →  http://localhost:3000
```

Mount trained model weights:
```bash
docker-compose up --build -v ./models:/neurovision/models:ro
```

---

## 📚 References

| Paper | Link |
|---|---|
| EfficientNet (Tan & Le, 2019) | [arxiv:1905.11946](https://arxiv.org/abs/1905.11946) |
| U-Net (Ronneberger et al., 2015) | [arxiv:1505.04597](https://arxiv.org/abs/1505.04597) |
| Attention U-Net (Oktay et al., 2018) | [arxiv:1804.03999](https://arxiv.org/abs/1804.03999) |
| Grad-CAM (Selvaraju et al., 2017) | [arxiv:1610.02391](https://arxiv.org/abs/1610.02391) |
| SmoothGrad (Smilkov et al., 2017) | [arxiv:1706.03825](https://arxiv.org/abs/1706.03825) |
| Integrated Gradients (Sundararajan et al., 2017) | [arxiv:1703.01365](https://arxiv.org/abs/1703.01365) |

---

## 📄 License

Distributed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built by [Dhivya Shri](https://github.com/DhivyaShri1385)**

*NeuroVision AI — Research Platform · Not for clinical use*

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![React](https://img.shields.io/badge/React-19-cyan?style=flat-square&logo=react)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-teal?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)

</div>
