# Streamlit Interactive Web Application Guide

## 1. Purpose & Capabilities
The project provides an interactive web application built with **Streamlit** ([`app.py`](../app.py)) that allows researchers and clinicians to test the trained model interactively.

### Key Capabilities:
1. **Interactive Image Upload:** Drag-and-drop or upload custom dermoscopic skin lesion images (JPEG / PNG).
2. **Preset Sample Testing:** Instant selection of curated cases from the validation dataset (`MEL`, `NV`, `BCC`, `OTHER`).
3. **Dual Model Mode:**
   - **4-Class Mode [Default]:** `NV`, `MEL`, `BCC`, `OTHER` (`models/best_resnet50_4class.pth`).
   - **8-Class Mode [Legacy]:** Full 8-class ISIC benchmark (`models/best_resnet50.pth`).
4. **Probability Distribution:** Visual horizontal/vertical bar charts of softmax prediction probabilities.
5. **Real-Time Grad-CAM Saliency:** Live 3-panel display of Original Image, Saliency Heatmap, and Blended Overlay.
6. **Built-in Performance Dashboard:** Integrated inspection of training curves, confusion matrices, and EDA plots.

---

## 2. Launching the Application

Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

Start the Streamlit application:
```bash
streamlit run app.py
```

By default, the server will launch locally at:
```text
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

---

## 3. UI Layout Overview

```text
┌────────────────────────────────────────────────────────────────────────┐
│  🔬 ISIC 2019 Dermoscopic Skin Lesion Classifier                       │
├────────────────────────────────┬───────────────────────────────────────┤
│ Sidebar:                       │ Tab 1: Upload & Classify              │
│ - Model Mode Selector          │ ┌───────────────────┬───────────────┐ │
│   (4-Class / 8-Class)          │ │ Upload / Sample   │ Confidence %  │ │
│ - Checkpoint File Path         │ │ Image Display     │ Bar Chart     │ │
│ - Supported Classes Key        │ └───────────────────┴───────────────┘ │
│                                │ ─── Grad-CAM Explainability ────────  │
│                                │ [Original]   [Heatmap]   [Overlay]   │
│                                ├───────────────────────────────────────┤
│                                │ Tab 2: Model Evaluation & Metrics     │
│                                │ [Loss/Acc Curves]  [Confusion Matrix] │
│                                │ [EDA Distribution] [Test Performance] │
└────────────────────────────────┴───────────────────────────────────────┘
```

---

## 4. Key Application Files
- Main App script: [`app.py`](../app.py)
- Model loader & checkpoint reader: [`src/model.py`](../src/model.py), [`src/train_4class.py`](../src/train_4class.py)
- Explainability module: [`src/gradcam.py`](../src/gradcam.py)
