# Deep Learning-Based Multiclass Skin Lesion Classification Using ISIC 2019
### *An Image Classification Approach Using Transfer Learning and Explainable AI (Grad-CAM)*

---

## 1. Project Objective & Motivation

This project implements an end-to-end deep learning image classification system for dermatological diagnosis using the **ISIC 2019 Challenge dataset** (25,331 training dermoscopic images and 8,238 test images).

### 4-Class Diagnostic Target
To address the severe class imbalance of rare diagnostic categories in the original 8-class dataset without discarding valuable clinical training data, the classification target maps all available samples into **4 primary clinical categories**:

1. **`NV` (Melanocytic Nevus):** Benign moles (Class index: `0`).
2. **`MEL` (Melanoma):** High-risk malignant skin cancer (Class index: `1`).
3. **`BCC` (Basal Cell Carcinoma):** Common malignant non-melanoma skin cancer (Class index: `2`).
4. **`OTHER` (Other Diagnoses):** Aggregates underrepresented clinical categories into a single group (Class index: `3`):
   - *Actinic Keratosis (`AK`)*
   - *Benign Keratosis (`BKL`)*
   - *Dermatofibroma (`DF`)*
   - *Vascular Lesion (`VASC`)*
   - *Squamous Cell Carcinoma (`SCC`)*

$$\text{OTHER} = \text{AK} + \text{BKL} + \text{DF} + \text{VASC} + \text{SCC} = 4,611 \text{ images}$$

> **Key Advantage:** All **25,331 images** remain actively utilized in training. Grouping minority classes creates a well-represented 4-class problem, boosting model accuracy to **71.33% on validation** and **66.74% on the official test set** (compared to 57.12% and 50.06% on the 8-class baseline).

---

## 2. Pipeline Architecture

```text
                 ISIC 2019 Dataset (25,331 images)
                                │
                                ▼
                    Basic EDA & 4-Class Mapping
               (NV=0, MEL=1, BCC=2, OTHER=3)
                                │
                                ▼
              Lesion-Safe Stratified Split (80/20)
                  (Zero Patient Lesion Leakage)
                                │
                                ▼
              Preprocessing & Moderate Augmentation
            (Resize 224x224, Flips, Rotations, Jitter)
                                │
                                ▼
                     Pretrained ResNet-50
                                │
                                ▼
                 4-Class Weighted Cross Entropy
                  (Computed from Training Split)
                                │
                                ▼
                   Model Training & Fine-Tuning
                 (MPS / CUDA / CPU Acceleration)
                                │
                                ▼
                    Evaluation & Explainability
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
      Performance Metrics                      Grad-CAM
  (Accuracy, Macro F1, Confusion Mat)    (ResNet-50 Layer 4 Saliency)
              │                                   │
              └─────────────────┬─────────────────┘
                                ▼
                    Interactive Streamlit App
```

---

## 3. Project Structure

```text
.
├── data/
│   ├── raw/
│   │   ├── train/ISIC_2019_Training_Input/
│   │   └── test/ISIC_2019_Test_Input/
│   └── processed/
│       ├── train_split_4class.csv   (20,297 images, 80.13%)
│       ├── val_split_4class.csv     (5,034 images, 19.87%)
│       ├── test_split_4class.csv    (6,191 images)
│       ├── train_split.csv          (8-class legacy)
│       └── val_split.csv            (8-class legacy)
├── models/
│   ├── resnet50_imagenet.pth        (Base ImageNet backbone)
│   ├── best_resnet50_4class.pth     (Primary 4-Class trained checkpoint)
│   └── best_resnet50.pth            (Legacy 8-Class checkpoint)
├── results/
│   ├── plots/
│   │   ├── eda_4class_distribution.png
│   │   ├── training_curves_4class.png
│   │   ├── confusion_matrix_validation_4class.png
│   │   ├── confusion_matrix_test_4class.png
│   │   ├── eda_class_distribution.png (8-class)
│   │   ├── training_curves.png (8-class)
│   │   └── confusion_matrix_validation.png (8-class)
│   ├── metrics/
│   │   ├── training_history_4class.json
│   │   ├── metrics_validation_4class.json
│   │   ├── metrics_test_4class.json
│   │   └── predictions_validation_4class.csv
│   └── gradcam/
│       ├── gradcam_4class_summary_grid.png
│       └── gradcam_summary_grid.png
├── src/
│   ├── dataset_4class.py            (4-class label mapping & lesion-safe splitting)
│   ├── train_4class.py              (4-class training & fine-tuning engine)
│   ├── evaluate_4class.py           (4-class validation & test evaluation)
│   ├── gradcam_4class.py            (4-class Grad-CAM explainability generator)
│   ├── dataset.py                   (FastBatchLoader & image augmentations)
│   ├── model.py                     (ResNet-50 definition & loss functions)
│   ├── metrics.py                   (Accuracy, Macro F1, Weighted F1 calculators)
│   └── gradcam.py                   (Grad-CAM hook engine)
├── memory/
│   ├── project_architecture_memory.md
│   └── pytorch_macos_troubleshooting.md
├── app.py                           (Streamlit interactive web app)
├── requirements.txt
└── README.md
```

---

## 4. Performance & Evaluation Results

### Comparison: 4-Class Model vs. 8-Class Baseline

| Model Configuration | Validation Accuracy | Validation Macro F1 | Test Accuracy | Test Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **4-Class ResNet-50 (NV, MEL, BCC, OTHER)** | **71.33%** | **0.6522** | **66.74%** | **0.6394** |
| 8-Class ResNet-50 (Baseline) | 57.12% | 0.3579 | 50.06% | 0.3267 |
| **Improvement** | **+14.21%** | **+0.2943** | **+16.68%** | **+0.3127** |

---

### Detailed 4-Class Validation Set Performance (5,034 images)

| Class | Diagnosis Name | Precision | Recall (Sensitivity) | F1-Score | Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **NV** | Melanocytic Nevus | **85.96%** | **79.96%** | **0.8285** | 2,734 |
| **MEL** | Melanoma | **51.09%** | **58.06%** | **0.5435** | 844 |
| **BCC** | Basal Cell Carcinoma | **59.68%** | **76.94%** | **0.6722** | 581 |
| **OTHER** | Other Lesions (AK, BKL, DF, VASC, SCC) | **59.77%** | **53.49%** | **0.5645** | 875 |
| **Overall** | **Summary** | **Accuracy: 71.33%** | **Macro F1: 0.6522** | **Weighted F1: 0.7168** | **5,034** |

---

### Detailed 4-Class Official ISIC 2019 Test Set Performance (6,191 images)

*Note: The official test labels were mapped into the 4 project categories.*

| Class | Diagnosis Name | Precision | Recall (Sensitivity) | F1-Score | Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **NV** | Melanocytic Nevus | **82.84%** | **76.23%** | **0.7940** | 2,495 |
| **MEL** | Melanoma | **61.30%** | **64.81%** | **0.6300** | 1,327 |
| **BCC** | Basal Cell Carcinoma | **54.47%** | **74.36%** | **0.6288** | 975 |
| **OTHER** | Other Lesions (AK, BKL, DF, VASC, SCC) | **55.56%** | **46.27%** | **0.5049** | 1,394 |
| **Overall** | **Summary** | **Accuracy: 66.74%** | **Macro F1: 0.6394** | **Weighted F1: 0.6677** | **6,191** |

---

## 5. Quickstart & Execution Commands

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare 4-Class Dataset & Stratified Splits
```bash
python3 src/dataset_4class.py
```

### 3. Train 4-Class ResNet-50
```bash
python3 src/train_4class.py --epochs 5 --batch_size 32 --lr 2e-4
```

### 4. Evaluate Validation and Official Test Sets
```bash
python3 src/evaluate_4class.py
```

### 5. Generate Grad-CAM Explainability Visualizations
```bash
python3 src/gradcam_4class.py
```

### 6. Launch the Interactive Web Application
```bash
streamlit run app.py
```

---

## 6. Explainable AI: Grad-CAM Interpretation

Grad-CAM visual heatmaps are extracted from ResNet-50's final convolutional layer (`layer4`):
- **Melanoma (`MEL`):** Focuses on asymmetric pigment networks, peripheral dots, and irregular multi-color borders.
- **Basal Cell Carcinoma (`BCC`):** Highlights arborizing telangiectasia (vascular branching) and shiny white structures.
- **Nevus (`NV`):** Centers symmetrically on uniform, reticular pigment patterns.
- **Other (`OTHER`):** Highlights hyperkeratotic scaling and vascular blushes.

Visual summary grids are saved in [`results/gradcam/gradcam_4class_summary_grid.png`](results/gradcam/gradcam_4class_summary_grid.png).
