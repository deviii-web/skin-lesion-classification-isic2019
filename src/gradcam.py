"""
Phase 10: Explainable AI - Grad-CAM Implementation for ResNet-50
Generates Class Activation Maps (CAM) highlighting the visual regions influencing model decisions.
Produces 3-panel visualizations: Original Image | Grad-CAM Heatmap | Overlay.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import CLASSES, CLASS_NAMES, NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD
from src.model import build_resnet50, get_device

class GradCAM:
    """
    Grad-CAM implementation tailored for ResNet-50 layer4 convolutional features.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Computes Grad-CAM heatmap for a single image tensor of shape [1, 3, H, W].
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Target class score
        score = output[0, target_class]
        score.backward()

        # Global average pooling of gradients
        # gradients: [1, C, H, W] -> weights: [C]
        weights = torch.mean(self.gradients[0], dim=(1, 2))

        # Linear combination of feature maps
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=self.activations.device)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i, :, :]

        # Apply ReLU to retain only positive influence
        cam = F.relu(cam)
        cam = cam.cpu().numpy()

        # Normalize between 0 and 1
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, float(torch.softmax(output, dim=1)[0, target_class].item())

def apply_heatmap_overlay(img_pil, heatmap, colormap='jet', alpha=0.45):
    """
    Resizes heatmap to match original image and creates visual overlay.
    """
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(img_pil.size, Image.Resampling.BILINEAR)
    heatmap_np = np.array(heatmap_resized) / 255.0

    cmap = plt.get_cmap(colormap)
    colored_heatmap = cmap(heatmap_np)[:, :, :3]  # drop alpha channel

    img_np = np.array(img_pil.convert('RGB')) / 255.0
    overlay = (1 - alpha) * img_np + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 1)

    return heatmap_np, overlay

def run_gradcam_analysis(
    model_path="models/best_resnet50.pth",
    predictions_csv="results/metrics/predictions_validation.csv",
    output_dir="results/gradcam",
    num_samples=6
):
    print("=" * 60)
    print("PHASE 10: Grad-CAM Explainability Analysis")
    print("=" * 60)

    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Model
    print(f"Loading model checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = build_resnet50(num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Attach Grad-CAM to layer4
    grad_cam = GradCAM(model, model.layer4)

    # 3. Load predictions & images
    preds_df = pd.read_csv(predictions_csv)
    raw_val_df = pd.read_csv("data/processed/val_split.csv")
    merged_df = pd.merge(preds_df, raw_val_df[['image_id', 'image_path']], on='image_id', how='left')

    # Select representative samples:
    # 1. High-confidence Correct Melanoma (MEL)
    # 2. High-confidence Correct Nevus (NV)
    # 3. High-confidence Correct Basal Cell Carcinoma (BCC)
    # 4. Correct Vascular Lesion (VASC)
    # 5. Misclassified sample
    # 6. Another challenging misclassified sample
    sample_records = []

    correct_mel = merged_df[(merged_df['correct']) & (merged_df['true_label'] == 'MEL')].sort_values(by='confidence', ascending=False).head(1)
    correct_nv = merged_df[(merged_df['correct']) & (merged_df['true_label'] == 'NV')].sort_values(by='confidence', ascending=False).head(1)
    correct_bcc = merged_df[(merged_df['correct']) & (merged_df['true_label'] == 'BCC')].sort_values(by='confidence', ascending=False).head(1)
    correct_vasc = merged_df[(merged_df['correct']) & (merged_df['true_label'] == 'VASC')].sort_values(by='confidence', ascending=False).head(1)
    incorrect_samples = merged_df[~merged_df['correct']].sort_values(by='confidence', ascending=False).head(2)

    sample_df = pd.concat([correct_mel, correct_nv, correct_bcc, correct_vasc, incorrect_samples]).reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    print(f"\nGenerating Grad-CAM for {len(sample_df)} representative cases...")

    fig, axes = plt.subplots(len(sample_df), 3, figsize=(13, 3.8 * len(sample_df)))

    for idx, row in sample_df.iterrows():
        img_pil = Image.open(row['image_path']).convert('RGB')
        input_tensor = transform(img_pil).unsqueeze(0).to(device)

        # Generate CAM for predicted class
        heatmap, pred_idx, conf = grad_cam.generate(input_tensor, target_class=row['pred_label_idx'])
        _, overlay = apply_heatmap_overlay(img_pil, heatmap)

        # Column 1: Original Image
        ax1 = axes[idx, 0]
        ax1.imshow(img_pil)
        ax1.axis('off')
        ax1.set_title(f"Image: {row['image_id']}\nTrue: {row['true_label']} ({CLASS_NAMES[row['true_label']]})", fontsize=11, fontweight='bold')

        # Column 2: Heatmap
        ax2 = axes[idx, 1]
        im2 = ax2.imshow(heatmap, cmap='jet')
        ax2.axis('off')
        ax2.set_title(f"Grad-CAM Heatmap (layer4)", fontsize=11, fontweight='bold')

        # Column 3: Overlay
        ax3 = axes[idx, 2]
        ax3.imshow(overlay)
        ax3.axis('off')
        status = "CORRECT" if row['correct'] else "MISCLASSIFIED"
        color = "green" if row['correct'] else "red"
        ax3.set_title(f"Prediction: {row['pred_label']} ({conf*100:.1f}% conf)\nStatus: {status}", fontsize=11, fontweight='bold', color=color)

        # Save individual figure
        indiv_path = os.path.join(output_dir, f"gradcam_{row['image_id']}_{row['pred_label']}.png")
        fig_indiv, ax_i = plt.subplots(1, 3, figsize=(12, 4))
        ax_i[0].imshow(img_pil)
        ax_i[0].set_title(f"Original ({row['true_label']})")
        ax_i[0].axis('off')
        ax_i[1].imshow(heatmap, cmap='jet')
        ax_i[1].set_title("Heatmap")
        ax_i[1].axis('off')
        ax_i[2].imshow(overlay)
        ax_i[2].set_title(f"Pred: {row['pred_label']} ({conf*100:.1f}%)")
        ax_i[2].axis('off')
        fig_indiv.tight_layout()
        fig_indiv.savefig(indiv_path, dpi=200)
        plt.close(fig_indiv)

    plt.suptitle("ISIC 2019 ResNet-50 Grad-CAM Visual Explainability (Layer 4)", fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    summary_plot_path = os.path.join(output_dir, "gradcam_summary_grid.png")
    plt.savefig(summary_plot_path, dpi=300)
    plt.close()
    print(f"Saved Grad-CAM summary grid: {summary_plot_path}")
    print(f"Saved individual Grad-CAM case files in: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    run_gradcam_analysis()
