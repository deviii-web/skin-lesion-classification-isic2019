# Dataset Preparation, Label Mapping & Exploratory Data Analysis

## 1. Dataset Origin & Overview
The dataset used in this project originates from the **ISIC 2019: Skin Lesion Analysis Towards Melanoma Detection** challenge hosted by the International Skin Imaging Collaboration (ISIC).

- **Total Training Images:** 25,331 JPEG dermoscopic images.
- **Total Test Images:** 8,238 JPEG dermoscopic images.
- **Diagnostic Modality:** High-resolution dermoscopy.

---

## 2. Diagnostic Categories & Class Mappings

### A. Original 8-Class Scheme
1. **`MEL`** — Melanoma (4,522 images, 17.85%): Malignant neoplasm originating in melanocytes.
2. **`NV`** — Melanocytic Nevus (12,875 images, 50.83%): Benign proliferation of melanocytes.
3. **`BCC`** — Basal Cell Carcinoma (3,323 images, 13.12%): Common malignant epithelial tumor.
4. **`AK`** — Actinic Keratosis (867 images, 3.42%): Pre-cancerous epidermal lesion.
5. **`BKL`** — Benign Keratosis (2,624 images, 10.36%): Seborrheic keratosis, solar lentigo, and lichen-planus like keratosis.
6. **`DF`** — Dermatofibroma (239 images, 0.94%): Benign fibrohistiocytic skin lesion.
7. **`VASC`** — Vascular Lesion (253 images, 1.00%): Angiomas, angiokeratomas, pyogenic granulomas.
8. **`SCC`** — Squamous Cell Carcinoma (628 images, 2.48%): Invasive malignant epithelial neoplasm.

### B. 4-Class Mapping Scheme
To mitigate extreme class imbalance (where `DF` and `VASC` account for <1% each), the 5 rare/underrepresented categories were aggregated into an `OTHER` group while keeping all 25,331 images:

```python
LABEL_MAP = {
    'NV': 'NV',        # Class 0: Melanocytic Nevus (12,875 images, 50.83%)
    'MEL': 'MEL',      # Class 1: Melanoma (4,522 images, 17.85%)
    'BCC': 'BCC',      # Class 2: Basal Cell Carcinoma (3,323 images, 13.12%)
    'AK': 'OTHER',     # Class 3: Other Diagnoses (Combined: 4,611 images, 18.20%)
    'BKL': 'OTHER',
    'DF': 'OTHER',
    'VASC': 'OTHER',
    'SCC': 'OTHER'
}
```

```text
OTHER = AK (867) + BKL (2,624) + DF (239) + VASC (253) + SCC (628) = 4,611 images
```

---

## 3. Data Integrity & Validation Pipeline
The validation script ([`src/dataset_validation.py`](../src/dataset_validation.py)) verifies:
- Ground truth row count: 25,331 rows.
- Duplicate Image IDs: 0.
- Missing image files: 0.
- Corrupted image files: 0.
- One-hot sum validation: exactly 1.0 per image record.

---

## 4. Patient-Leakage Prevention & Stratified Splitting
A critical requirement in clinical image classification is preventing **data leakage** across patient partitions. Since multiple dermoscopic images may depict the same lesion across follow-up visits, standard random splitting can place images from the same lesion in both train and validation sets.

### Splitting Methodology:
- Algorithm: `StratifiedGroupKFold` ($k=5$, 80% train / 20% validation).
- Grouping Column: `lesion_id` (13,931 unique lesions).
- **Result:**
  - **4-Class Train Split:** 20,297 images (80.13%) → [`data/processed/train_split_4class.csv`](../data/processed/train_split_4class.csv)
  - **4-Class Validation Split:** 5,034 images (19.87%) → [`data/processed/val_split_4class.csv`](../data/processed/val_split_4class.csv)
  - **Lesion ID Overlap:** **Strictly 0**.

---

## 5. Exploratory Data Analysis (EDA) Findings

Visual artifacts are generated in [`results/plots/`](../results/plots/):

1. **4-Class Diagnostic Distribution ([`results/plots/eda_4class_distribution.png`](../results/plots/eda_4class_distribution.png)):**
   - Shows the balanced 4-way partition: `NV` (50.0%), `OTHER` (18.4%), `MEL` (18.1%), `BCC` (13.5%).
2. **8-Class Distribution ([`results/plots/eda_class_distribution.png`](../results/plots/eda_class_distribution.png)):**
   - Documents the severe long-tail distribution of the original challenge.
3. **Multi-Class Image Samples ([`results/plots/eda_sample_images.png`](../results/plots/eda_sample_images.png)):**
   - Visualizes representative dermoscopic appearances across all lesion types.
4. **Image Dimensions & Aspect Ratios ([`results/plots/eda_image_dimensions.png`](../results/plots/eda_image_dimensions.png)):**
   - Resolutions range from 600×450 to 1024×1024 px with mean aspect ratio ~1.19. Standardized to 224×224 px for CNN ingestion.
