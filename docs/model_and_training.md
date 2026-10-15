# Model Architecture, Class Weighting & Training Pipeline

## 1. Network Architecture: ResNet-50
The model uses a **ResNet-50 (Residual Network 50-layer)** backbone initialized with **ImageNet** pretrained weights (`ResNet50_Weights.DEFAULT`).

### Architectural Topology:
```text
Input Dermoscopic Image (3 × 224 × 224)
                │
                ▼
      Conv1 (7×7, 64, stride 2) + MaxPool
                │
                ▼
      Layer 1 (3 Residual Bottleneck Blocks)
                │
                ▼
      Layer 2 (4 Residual Bottleneck Blocks)
                │
                ▼
      Layer 3 (6 Residual Bottleneck Blocks)
                │
                ▼
      Layer 4 (3 Residual Bottleneck Blocks, 2048-dim features) ──► [Grad-CAM Hook Point]
                │
                ▼
       Adaptive Average Pooling (2048-dim vector)
                │
                ▼
       Fully Connected Linear Head (2048 ──► Num_Classes)
                │
                ▼
          Logits / Softmax Class Probabilities
```

- **4-Class Head:** `nn.Linear(2048, 4)` for `NV`, `MEL`, `BCC`, `OTHER`.
- **8-Class Head:** `nn.Linear(2048, 8)` for full legacy benchmark.

---

## 2. Image Preprocessing & Data Augmentation

To promote generalization without introducing unrealistic clinical artifacts, a moderate augmentation pipeline was applied:

### Training Pipeline:
1. **Resize:** Standardized to $224 \times 224$ pixels.
2. **Random Horizontal Flip:** $p=0.5$ (simulates dermatoscope orientation invariance).
3. **Random Vertical Flip:** $p=0.5$.
4. **Random Rotation:** $\pm 15^\circ$.
5. **Color Jitter:** Minor adjustments ($\text{brightness}=0.15, \text{contrast}=0.15, \text{saturation}=0.1$) to account for variable lighting.
6. **Normalization:** ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

### Validation / Test Pipeline:
1. **Resize:** $224 \times 224$ pixels.
2. **Normalization:** ImageNet statistics (deterministic, no random perturbations).

---

## 3. Class Imbalance Mitigation

To prevent the dominant `NV` class (~50%) from overwhelming gradient updates, **balanced inverse-frequency loss weighting** was calculated **strictly from the training partition**:

$$w_c = \frac{N}{C \cdot N_c}$$

Normalized so that $\bar{w} = 1.0$:

```text
Class 0 (NV):     0.4054 (Down-weighted due to high frequency)
Class 1 (MEL):    1.1190 (Penalized higher to protect against missed malignancies)
Class 2 (BCC):    1.4138 (Penalized higher)
Class 3 (OTHER):  1.0618
```

---

## 4. In-Process FastBatchLoader

To guarantee high stability across operating systems (specifically resolving POSIX multi-worker subprocess fork segmentation faults on macOS Python 3.13), a custom in-process batch iterator (`FastBatchLoader`) was implemented:
- Single-threaded, zero-IPC batch generation.
- Batch shuffling and vector tensor stacking directly into PyTorch device memory.

---

## 5. Training Mechanics & Hyperparameters

- **Optimizer:** `AdamW` (weight decay $\lambda = 10^{-4}$ for L2 regularization).
- **Learning Rate:** Initial $\eta = 2 \times 10^{-4}$.
- **Learning Rate Schedule:** `StepLR` (step size = 2 epochs, $\gamma = 0.5$).
- **Batch Size:** 32.
- **Compute Accelerator:** Metal Performance Shaders (`mps` on Apple Silicon), CUDA GPU, or CPU fallback.
- **Checkpointing Policy:** Automatically saves the state dict with the highest validation performance to `models/best_resnet50_4class.pth`.
