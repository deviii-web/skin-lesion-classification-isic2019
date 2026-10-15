# Explainable AI: Grad-CAM Saliency Analysis

## 1. What is Grad-CAM?
**Grad-CAM (Gradient-weighted Class Activation Mapping)** is a visual explanation technique for convolutional neural networks (CNNs). It produces a coarse 2D heatmap highlighting the discriminative regions in an input image that contributed most heavily to a particular class prediction.

---

## 2. Mathematical Formulation

Let $A^k$ denote the $k$-th feature activation map of the target convolutional layer (in our case, the final residual bottleneck block `layer4` of ResNet-50), and let $y^c$ denote the unnormalized score (logit) for class $c$.

### Step 1: Neuron Importance Weights ($\alpha_k^c$)
Gradients of the score $y^c$ with respect to feature activation maps $A^k$ are computed and globally average-pooled over spatial dimensions $(i, j)$:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i, j}^k}$$

where $Z = H \times W$ is the spatial area of the feature map.

### Step 2: Linear Combination & Rectification
A weighted sum of all feature maps is computed followed by a Rectified Linear Unit ($\text{ReLU}$) to isolate features that have a positive influence on the target diagnosis:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

---

## 3. Implementation Details ([`src/gradcam.py`](../src/gradcam.py))

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # PyTorch forward and full backward hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        score.backward()

        weights = torch.mean(self.gradients[0], dim=(1, 2))
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=self.activations.device)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i, :, :]

        cam = F.relu(cam).cpu().numpy()
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        return cam, target_class, float(torch.softmax(output, dim=1)[0, target_class].item())
```

---

## 4. Clinical Interpretation of Saliency Maps

Grad-CAM heatmaps generated for representative samples ([`results/gradcam/gradcam_4class_summary_grid.png`](../results/gradcam/gradcam_4class_summary_grid.png)) reveal distinct morphological patterns:

1. **Melanoma (`MEL`):**
   - The model highlights peripheral irregular borders, asymmetrical pigment networks, and focal dark streaks characteristic of invasive melanocytic lesions.
2. **Basal Cell Carcinoma (`BCC`):**
   - Saliency focuses on shiny white structures, translucency, and arborizing telangiectatic vessels.
3. **Melanocytic Nevus (`NV`):**
   - The heatmap displays homogenous, centralized attention over regular reticular pigment patterns.
4. **Other Lesions (`OTHER`):**
   - Saliency tracks hyperkeratotic scale (in `AK`/`BKL`) or localized vascular lacunae (in `VASC`).

---

## 5. Scope & Caveat
Grad-CAM heatmaps illustrate **what pixel regions influenced model activation**. They do not guarantee clinical correctness and must be utilized strictly as assistive visual tools alongside licensed dermatological evaluation.
