# NeuroVision AI
### Explainable Brain Tumor Diagnosis & Segmentation Platform

> A production-quality AI healthcare platform for classifying and segmenting brain MRI scans with full explainability — Grad-CAM, saliency maps, and clinical PDF reports.

---

## Overview

NeuroVision AI is an end-to-end deep learning platform that:

- **Classifies** brain MRI scans into 4 categories: Glioma, Meningioma, Pituitary, No Tumor
- **Segments** tumor regions using Attention U-Net
- **Explains** predictions via Grad-CAM heatmaps and saliency maps
- **Reports** findings in clinical-style PDF reports
- **Serves** predictions through a FastAPI REST backend
- **Displays** results on a React + Tailwind CSS dark-theme dashboard

---

## Tech Stack

| Layer | Technology |
|---|---|
| Classification | EfficientNetB3 + ResNet50 + DenseNet121 (Ensemble) |
| Segmentation | Attention U-Net (MONAI) |
| Explainability | Grad-CAM, Saliency Maps, Integrated Gradients |
| Backend | FastAPI + PostgreSQL |
| Frontend | React + Tailwind CSS |
| Training | TensorFlow 2.x + Albumentations |
| Deployment | Docker + Render / HuggingFace Spaces |

---

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — 4 balanced classes:

| Split | Glioma | Meningioma | No Tumor | Pituitary | Total |
|---|---|---|---|---|---|
| Train | 335 | 335 | 335 | 335 | 1,340 |
| Val   | 99  | 99  | 99  | 99  | 396   |
| Test  | 49  | 49  | 49  | 49  | 196   |
| **Total** | **483** | **483** | **483** | **483** | **1,932** |

---

## Project Structure

```
neurovision-ai/
├── configs/
│   └── config.yaml              # All hyperparameters (single source of truth)
├── src/
│   ├── data/
│   │   ├── dataset_organizer.py # Audit & integrity check
│   │   ├── preprocessor.py      # Resize, normalize, grayscale→RGB
│   │   ├── augmentation.py      # Albumentations medical pipeline
│   │   ├── splitter.py          # Split validation & CSV manifests
│   │   └── data_loader.py       # tf.data pipeline (AUTOTUNE, prefetch)
│   ├── models/                  # Phase 2 — classifiers & ensemble
│   ├── segmentation/            # Phase 3 — Attention U-Net
│   ├── xai/                     # Phase 4 — Grad-CAM, saliency
│   ├── visualization/
│   │   └── eda_visualizer.py    # EDA plots (dark theme)
│   └── utils/
│       ├── logger.py            # Loguru structured logging
│       └── config.py            # Typed frozen config dataclass
├── scripts/
│   ├── run_phase1.py            # Dataset audit & preprocessing
│   ├── run_phase2.py            # Model training (Phase 2)
│   └── run_phase3.py            # Segmentation training (Phase 3)
├── api/                         # Phase 5 — FastAPI backend
├── frontend/                    # Phase 6 — React dashboard
├── outputs/
│   ├── manifest_train.csv       # Train split manifest
│   ├── manifest_val.csv         # Val split manifest
│   └── manifest_test.csv        # Test split manifest
├── requirements.txt
├── .env.example
└── docker-compose.yml           # Phase 7
```

---

## Development Phases

| Phase | Description | Status |
|---|---|---|
| **1** | Dataset organization, preprocessing, augmentation, EDA | ✅ Complete |
| **2** | EfficientNetB3 training, ensemble, evaluation metrics | 🔄 In Progress |
| **3** | Attention U-Net segmentation, Dice score | ⏳ Planned |
| **4** | Grad-CAM, saliency maps, explainability dashboard | ⏳ Planned |
| **5** | FastAPI backend, REST endpoints, model serving | ⏳ Planned |
| **6** | React frontend, upload UI, visualization dashboard | ⏳ Planned |
| **7** | PDF reports, Docker, deployment (Render / HuggingFace) | ⏳ Planned |

---

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/DhivyaShri1385/NeuroVision-AI.git
cd NeuroVision-AI/neurovision-ai
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env — set DATASET_ROOT to your dataset path
```

### 3. Run Phase 1 (Dataset Audit & Preprocessing)
```bash
python scripts/run_phase1.py
```

Expected output:
```
PHASE 1 COMPLETE
  Total images   : 1932
  Classes        : ['glioma', 'meningioma', 'no_tumor', 'pituitary']
  Train samples  : 1340
  Val samples    : 396
  Test samples   : 196
  Corrupted      : 0
```

---

## Model Architecture

### Classification — EfficientNetB3 + Ensemble

```
Input (224×224×3)
    │
EfficientNetB3 (pretrained ImageNet, frozen 10 epochs)
    │
GlobalAveragePooling2D
    │
Dense(512, ReLU) → Dropout(0.4)
    │
Dense(4, Softmax)  →  [Glioma | Meningioma | No Tumor | Pituitary]
```

Final ensemble: EfficientNetB3 + ResNet50 + DenseNet121 (soft voting).

### Segmentation — Attention U-Net

```
Input (256×256×1, grayscale)
    │
Encoder: [64 → 128 → 256 → 512 filters]
    │        ← Attention Gates
Decoder: [512 → 256 → 128 → 64]
    │
Output: Binary mask (tumor / no tumor)
```

---

## Training Details

| Parameter | Value |
|---|---|
| Backbone | EfficientNetB3 |
| Pretrained | ImageNet |
| Input size | 224 × 224 × 3 |
| Batch size | 32 |
| Epochs | 80 (early stopping patience=12) |
| Optimizer | Adam (lr=1e-3 → 1e-5 fine-tune) |
| Loss | Categorical Cross-Entropy (label smoothing=0.1) |
| Augmentation | Rotation ±15°, flip, elastic, brightness, CLAHE |

---

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score (per class + macro)
- ROC-AUC (one-vs-rest)
- Confusion Matrix
- Dice Score & IoU (segmentation)

---

## Explainability

| Method | Purpose |
|---|---|
| Grad-CAM | Highlights which pixels most influenced the prediction |
| Saliency Maps | Gradient of output w.r.t. input pixels |
| Integrated Gradients | Attribution of each feature to the output |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Dhivya Shri** — AI/ML Engineer  
[GitHub](https://github.com/DhivyaShri1385)
