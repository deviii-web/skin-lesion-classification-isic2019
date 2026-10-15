"""
4-Class Grad-CAM Explainability Analysis
Generates Grad-CAM visual heatmaps and overlays for the 4-class ResNet-50 model across:
- NV (Melanocytic Nevus)
- MEL (Melanoma)
- BCC (Basal Cell Carcinoma)
- OTHER (Actinic/Benign Keratosis, Dermatofibroma, Vascular, SCC)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_4class import CLASSES_4, CLASS_NAMES_4
from src.train_4class import build_4class_resnet50, get_4class_transforms
from src.gradcam import GradCAM, apply_heatmap_overlay
from src.model import get_device

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def run_4class_gradcam(
    model_path="models/best_resnet50_4class.pth",
    predictions_csv="results/metrics/predictions_validation_4class.csv",
    output_dir="results/gradcam"
):
    print("=" * 60)
    print("4-CLASS GRAD-CAM EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Model
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = build_4class_resnet50(pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    grad_cam = GradCAM(model, model.layer4)

    # 2. Load validation predictions & image paths
    preds_df = pd.read_csv(predictions_csv)
    raw_val_df = pd.read_csv("data/processed/val_split_4class.csv")
    merged_df = pd.merge(preds_df, raw_val_df[['image_id', 'image_path', 'original_label']], on='image_id', how='left')

    # Select representative samples from each of the 4 classes + misclassifications
    sample_records = []
    for cls in CLASSES_4:
        correct_sample = merged_df[(merged_df['correct']) & (merged_df['true_label'] == cls)].sort_values(by='confidence', ascending=False).head(1)
        sample_records.append(correct_sample)

    incorrect_samples = merged_df[~merged_df['correct']].sort_values(by='confidence', ascending=False).head(2)
    sample_records.append(incorrect_samples)
    samples_df = pd.concat(sample_records).reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    print(f"Generating Grad-CAM for {len(samples_df)} representative 4-class cases...")

    fig, axes = plt.subplots(len(samples_df), 3, figsize=(13, 3.8 * len(samples_df)))

    for idx, row in samples_df.iterrows():
        img_pil = Image.open(row['image_path']).convert('RGB')
        input_tensor = transform(img_pil).unsqueeze(0).to(device)

        heatmap, pred_idx, conf = grad_cam.generate(input_tensor, target_class=row['pred_label_idx'])
        _, overlay = apply_heatmap_overlay(img_pil, heatmap)

        # 1. Original Image
        ax1 = axes[idx, 0]
        ax1.imshow(img_pil)
        ax1.axis('off')
        orig_tag = f" ({row['original_label']})" if 'original_label' in row and pd.notna(row['original_label']) else ""
        ax1.set_title(f"Image: {row['image_id']}\nTrue: {row['true_label']}{orig_tag}", fontsize=11, fontweight='bold')

        # 2. Grad-CAM Heatmap
        ax2 = axes[idx, 1]
        ax2.imshow(heatmap, cmap='jet')
        ax2.axis('off')
        ax2.set_title(f"Grad-CAM Heatmap (Layer 4)", fontsize=11, fontweight='bold')

        # 3. Overlay
        ax3 = axes[idx, 2]
        ax3.imshow(overlay)
        ax3.axis('off')
        status = "CORRECT" if row['correct'] else "MISCLASSIFIED"
        color = "green" if row['correct'] else "red"
        ax3.set_title(f"Pred: {row['pred_label']} ({conf*100:.1f}% conf)\nStatus: {status}", fontsize=11, fontweight='bold', color=color)

        # Individual image file
        indiv_path = os.path.join(output_dir, f"gradcam_4class_{row['image_id']}_{row['pred_label']}.png")
        fig_indiv, ax_i = plt.subplots(1, 3, figsize=(12, 4))
        ax_i[0].imshow(img_pil)
        ax_i[0].set_title(f"True: {row['true_label']}")
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

    plt.suptitle("ISIC 2019 4-Class ResNet-50 Grad-CAM Explainability (NV, MEL, BCC, OTHER)", fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    summary_plot = os.path.join(output_dir, "gradcam_4class_summary_grid.png")
    plt.savefig(summary_plot, dpi=300)
    plt.close()
    print(f"Saved 4-Class Grad-CAM summary grid: {summary_plot}")
    print("=" * 60)

if __name__ == "__main__":
    run_4class_gradcam()
