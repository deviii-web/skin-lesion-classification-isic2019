# User Guide & CLI Execution Manual

## 1. System Requirements & Environment Setup

### Environment Prerequisites:
- **Operating System:** macOS (Apple Silicon arm64 / Intel), Linux (Ubuntu 20.04+), or Windows 10/11 with WSL2.
- **Python Version:** Python 3.10 to 3.13.
- **Hardware Acceleration:** Apple Silicon Metal Performance Shaders (`mps`), NVIDIA CUDA GPU (`cuda`), or Multi-core CPU (`cpu`).

### Installation:
```bash
# Clone or navigate to project directory
cd isic-classification

# Install Python dependencies
pip install -r requirements.txt
```

---

## 2. Step-by-Step Command Line Execution

All stages of the project are modular and can be run independently:

### Step 1: Dataset Validation & Preprocessing
Validates ground truth CSV integrity and image file presence:
```bash
python3 src/dataset_validation.py
```

### Step 2: Exploratory Data Analysis (EDA)
Generates the 8-class diagnostic distribution, aspect ratio scatter plots, and multi-class sample grids:
```bash
python3 src/eda.py
```

### Step 3: 4-Class Mapping & Lesion-Safe Splitting
Maps labels into `NV`, `MEL`, `BCC`, `OTHER` and generates patient-leakage safe 80/20 splits:
```bash
python3 src/dataset_4class.py
```

### Step 4: Train the 4-Class ResNet-50 Model
Trains the model on MPS/CUDA with weighted cross-entropy loss, step learning rate scheduling, and checkpointing:
```bash
python3 src/train_4class.py --epochs 5 --batch_size 32 --lr 2e-4
```

### Step 5: Evaluate 4-Class Model on Validation & Test Sets
Evaluates both the 5,034-sample validation split and the 6,191-sample official test set, computing accuracy, Macro F1, Weighted F1, and confusion matrices:
```bash
python3 src/evaluate_4class.py
```

### Step 6: Generate Grad-CAM Explainability Heatmaps
Computes layer4 saliency heatmaps across all 4 diagnostic classes:
```bash
python3 src/gradcam_4class.py
```

### Step 7: Launch the Web Demo Application
Starts the interactive browser interface:
```bash
streamlit run app.py
```

---

## 3. Directory Layout & File Reference

```text
isic-classification/
├── data/
│   ├── raw/
│   │   ├── train/ISIC_2019_Training_Input/ (25,331 images)
│   │   └── test/ISIC_2019_Test_Input/       (8,238 images)
│   └── processed/
│       ├── train_split_4class.csv
│       ├── val_split_4class.csv
│       └── test_split_4class.csv
├── models/
│   ├── resnet50_imagenet.pth
│   ├── best_resnet50_4class.pth (Primary 4-class checkpoint)
│   └── best_resnet50.pth        (Legacy 8-class checkpoint)
├── results/
│   ├── plots/
│   │   ├── eda_4class_distribution.png
│   │   ├── training_curves_4class.png
│   │   ├── confusion_matrix_validation_4class.png
│   │   └── confusion_matrix_test_4class.png
│   ├── metrics/
│   │   ├── metrics_validation_4class.json
│   │   └── metrics_test_4class.json
│   └── gradcam/
│       └── gradcam_4class_summary_grid.png
├── src/
│   ├── dataset_4class.py
│   ├── train_4class.py
│   ├── evaluate_4class.py
│   ├── gradcam_4class.py
│   ├── dataset.py
│   ├── model.py
│   └── metrics.py
├── docs/
│   ├── overview.md
│   ├── dataset_and_eda.md
│   ├── model_and_training.md
│   ├── evaluation_and_results.md
│   ├── explainability_gradcam.md
│   ├── streamlit_demo.md
│   └── user_guide.md
├── app.py
├── requirements.txt
└── README.md
```

---

## 4. Troubleshooting & Known Platform Solutions

### Issue: Multi-worker PyTorch `DataLoader` Segfault on macOS (`SIGSEGV` / `SIGBUS`)
- **Cause:** Python 3.13 POSIX multiprocessing fork/spawn behavior on macOS.
- **Solution:** Always use the built-in single-threaded [`FastBatchLoader`](../src/dataset.py:47) or set `num_workers=0`.

### Issue: Pretrained Model Loading Crash
- **Cause:** Direct `models.resnet50(weights=DEFAULT)` dynamic hook invocation in Python 3.13.
- **Solution:** Use the safe unweighted constructor with offline `.pth` loading as implemented in [`src/train_4class.py`](../src/train_4class.py:72).
