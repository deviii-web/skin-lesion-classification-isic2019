# Project Overview & Executive Summary

## 1. Project Title
**Deep Learning-Based Multiclass Skin Lesion Classification Using Dermoscopic Images**
*An Image Classification Approach Using Transfer Learning and Explainable AI (Grad-CAM)*

---

## 2. Executive Summary
This project provides an end-to-end deep learning framework for the automated classification of skin lesions from dermoscopic imagery using the **ISIC 2019 Challenge dataset** (comprising 25,331 training and 8,238 testing images).

The solution implements transfer learning via a **ResNet-50** architecture, balanced inverse-frequency class weighting to handle demographic and diagnostic class imbalances, and **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide visual explainability for clinical decision-making.

The project supports two evaluation regimes:
1. **Primary 4-Class Classification (`NV`, `MEL`, `BCC`, `OTHER`):** Groups severely underrepresented rare conditions into an aggregate `OTHER` class, achieving **71.33% validation accuracy** (0.6522 Macro F1) and **66.74% official test accuracy** (0.6394 Macro F1).
2. **Legacy 8-Class Benchmark:** Retains all 8 individual diagnostic categories (`MEL`, `NV`, `BCC`, `AK`, `BKL`, `DF`, `VASC`, `SCC`), establishing a baseline of 57.12% validation accuracy.

---

## 3. High-Level System Architecture

```text
                     ISIC 2019 Challenge Dataset
                                  │
                                  ▼
                 Dataset Validation & Integrity Checks
                                  │
                                  ▼
                    Exploratory Data Analysis (EDA)
                                  │
                                  ▼
                  4-Class / 8-Class Label Mapping
                                  │
                                  ▼
              Lesion-Safe Stratified Splitting (80/20)
                   (Zero Patient Lesion Leakage)
                                  │
                                  ▼
                Image Preprocessing & Augmentation
               (Resize 224x224, Flips, Rotations)
                                  │
                                  ▼
                  Transfer Learning (ResNet-50)
                                  │
                                  ▼
                  Weighted Cross-Entropy Loss
                                  │
                                  ▼
                     Model Training Engine
                  (MPS / CUDA / CPU Accelerated)
                                  │
                                  ▼
               Evaluation & Saliency Explainability
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
      Quantitative Metrics                   Grad-CAM Saliency Maps
  (Accuracy, Macro F1, CM)                (ResNet-50 Layer 4 Heatmaps)
              │                                       │
              └───────────────────┬───────────────────┘
                                  ▼
                     Streamlit Interactive Web App
```

---

## 4. Documentation Index

The `docs/` folder contains comprehensive documentation organized into dedicated modules:

- [`docs/overview.md`](docs/overview.md): Executive summary, scope, and architecture.
- [`docs/dataset_and_eda.md`](docs/dataset_and_eda.md): Dataset composition, class mapping, lesion-safe splitting, and EDA findings.
- [`docs/model_and_training.md`](docs/model_and_training.md): ResNet-50 architecture, class weighting, optimizer, and training mechanics.
- [`docs/evaluation_and_results.md`](docs/evaluation_and_results.md): Quantitative metrics, per-class breakdown, confusion matrices, and comparison tables.
- [`docs/explainability_gradcam.md`](docs/explainability_gradcam.md): Grad-CAM theory, hook implementation, and interpretation of saliency heatmaps.
- [`docs/streamlit_demo.md`](docs/streamlit_demo.md): Interactive application usage, layout, and deployment.
- [`docs/user_guide.md`](docs/user_guide.md): Step-by-step CLI execution guide, API reference, and troubleshooting.
