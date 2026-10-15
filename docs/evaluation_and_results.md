# Quantitative Evaluation & Empirical Results

## 1. Performance Overview: 4-Class vs. 8-Class

Aggregating the 5 severely underrepresented rare classes into `OTHER` creates a well-conditioned classification boundary while retaining 100% of the training images.

### Comprehensive Metric Comparison Table:

| Metric | 4-Class Model (Validation) | 4-Class Model (Official Test) | 8-Class Model (Validation) | 8-Class Model (Official Test) |
| :--- | :---: | :---: | :---: | :---: |
| **Top-1 Accuracy** | **71.33%** | **66.74%** | 57.12% | 50.06% |
| **Macro F1-Score** | **0.6522** | **0.6394** | 0.3579 | 0.3267 |
| **Weighted F1-Score** | **0.7168** | **0.6677** | 0.5911 | 0.5167 |
| **Evaluated Images** | 5,034 | 6,191 | 5,005 | 6,191 |

---

## 2. Detailed 4-Class Performance Breakdown

### A. Validation Set Results (5,034 samples)
Evaluated with [`data/processed/val_split_4class.csv`](../data/processed/val_split_4class.csv):

| Class Code | Diagnostic Category | Precision | Recall (Sensitivity) | F1-Score | True Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`NV`** | Melanocytic Nevus | **85.96%** | **79.96%** | **0.8285** | 2,734 |
| **`MEL`** | Melanoma | **51.09%** | **58.06%** | **0.5435** | 844 |
| **`BCC`** | Basal Cell Carcinoma | **59.68%** | **76.94%** | **0.6722** | 581 |
| **`OTHER`** | Other Lesions (AK, BKL, DF, VASC, SCC) | **59.77%** | **53.49%** | **0.5645** | 875 |
| **Summary** | **Overall Aggregate** | **71.33% Acc** | **0.6522 Macro F1** | **0.7168 Weighted F1** | **5,034** |

### 4-Class Validation Confusion Matrix:
- Rows = Ground Truth, Columns = Predictions

```text
               Pred NV    Pred MEL    Pred BCC    Pred OTHER
True NV         2186        174          76          298
True MEL         180        490          44          130
True BCC          34         39         447           61
True OTHER       143        256           8          468
```

---

### B. Official ISIC 2019 Test Set Results (6,191 samples)
Evaluated with [`data/processed/test_split_4class.csv`](../data/processed/test_split_4class.csv):

| Class Code | Diagnostic Category | Precision | Recall (Sensitivity) | F1-Score | True Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`NV`** | Melanocytic Nevus | **82.84%** | **76.23%** | **0.7940** | 2,495 |
| **`MEL`** | Melanoma | **61.30%** | **64.81%** | **0.6300** | 1,327 |
| **`BCC`** | Basal Cell Carcinoma | **54.47%** | **74.36%** | **0.6288** | 975 |
| **`OTHER`** | Other Lesions (AK, BKL, DF, VASC, SCC) | **55.56%** | **46.27%** | **0.5049** | 1,394 |
| **Summary** | **Overall Aggregate** | **66.74% Acc** | **0.6394 Macro F1** | **0.6677 Weighted F1** | **6,191** |

---

## 3. Clinical & Methodological Analysis

1. **High Malignancy Sensitivity (Recall on MEL and BCC):**
   - The model detects **64.81% of true melanomas** and **74.36% of basal cell carcinomas** on the unseen official test set.
   - High sensitivity on critical malignant categories is essential for clinical triage support.

2. **Strong Specificity on Benign Nevi (`NV`):**
   - Precision on benign moles remains high at **82.8%–86.0%**, minimizing unnecessary false-alarm biopsies on standard moles.

3. **Confusion Matrix Artifacts:**
   - Saved in [`results/plots/confusion_matrix_validation_4class.png`](../results/plots/confusion_matrix_validation_4class.png) and [`results/plots/confusion_matrix_test_4class.png`](../results/plots/confusion_matrix_test_4class.png).
   - Metrics saved in JSON format under [`results/metrics/`](../results/metrics/).
