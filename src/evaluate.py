"""
Phase 8: Comprehensive Model Evaluation Script
Computes:
- Overall Accuracy, Macro F1, Weighted F1
- Per-class Precision, Recall, F1-score, and Support
- Raw and Row-Normalized Confusion Matrices (saved to results/plots/)
- Complete JSON metrics summary (saved to results/metrics/)
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import FastBatchLoader, get_transforms, CLASSES, CLASS_NAMES, NUM_CLASSES
from src.model import build_resnet50, get_device
from src.metrics import compute_metrics

def evaluate_model(
    model_path="models/best_resnet50.pth",
    eval_csv="data/processed/val_split.csv",
    split_name="validation",
    output_plot_dir="results/plots",
    output_metrics_dir="results/metrics",
    batch_size=32
):
    print("=" * 60)
    print(f"PHASE 8: Model Evaluation ({split_name.upper()} SET)")
    print("=" * 60)

    device = get_device()
    os.makedirs(output_plot_dir, exist_ok=True)
    os.makedirs(output_metrics_dir, exist_ok=True)

    # 1. Load model checkpoint
    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = build_resnet50(num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Prepare Data
    eval_df = pd.read_csv(eval_csv)
    print(f"Evaluating {len(eval_df)} samples from {eval_csv}...")
    _, val_transform = get_transforms(224)
    loader = FastBatchLoader(eval_df, transform=val_transform, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []
    all_probs = []
    all_image_ids = []

    with torch.no_grad():
        for images, targets, ids in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            preds = outputs.argmax(dim=1).detach().cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_probs.extend(probs)
            all_image_ids.extend(ids)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    # 3. Compute Metrics
    acc, macro_f1, weighted_f1, per_class_f1s = compute_metrics(all_targets, all_preds, num_classes=NUM_CLASSES)

    # Per-class precision, recall, support
    per_class_metrics = {}
    print("\n" + "=" * 60)
    print(f"{'Class':<6} | {'Diagnosis':<25} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8} | {'Support':<8}")
    print("-" * 75)
    for c in range(NUM_CLASSES):
        cls_name = CLASSES[c]
        tp = np.sum((all_preds == c) & (all_targets == c))
        fp = np.sum((all_preds == c) & (all_targets != c))
        fn = np.sum((all_preds != c) & (all_targets == c))
        support = int(np.sum(all_targets == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = per_class_f1s[c]

        per_class_metrics[cls_name] = {
            'diagnosis': CLASS_NAMES[cls_name],
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'support': support
        }
        print(f"{cls_name:<6} | {CLASS_NAMES[cls_name]:<25} | {precision:<10.4f} | {recall:<8.4f} | {f1:<8.4f} | {support:<8}")

    print("-" * 75)
    print(f"Overall Accuracy:   {acc*100:.2f}%")
    print(f"Overall Macro F1:   {macro_f1:.4f}")
    print(f"Overall Weighted F1: {weighted_f1:.4f}")
    print("=" * 60)

    # 4. Confusion Matrix Computation
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    # 5. Plot Confusion Matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=CLASSES, yticklabels=CLASSES)
    ax1.set_title(f"Confusion Matrix (Raw Counts) - {split_name.title()}", fontweight='bold', fontsize=13)
    ax1.set_xlabel("Predicted Class", fontweight='bold')
    ax1.set_ylabel("True Ground Truth", fontweight='bold')

    # Normalized percentages
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', ax=ax2,
                xticklabels=CLASSES, yticklabels=CLASSES)
    ax2.set_title(f"Normalized Confusion Matrix - {split_name.title()}", fontweight='bold', fontsize=13)
    ax2.set_xlabel("Predicted Class", fontweight='bold')
    ax2.set_ylabel("True Ground Truth", fontweight='bold')

    plt.tight_layout()
    cm_plot_path = os.path.join(output_plot_dir, f"confusion_matrix_{split_name}.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"\nSaved confusion matrix plot: {cm_plot_path}")

    # 6. Save JSON Metrics
    metrics_summary = {
        'split': split_name,
        'total_samples': len(all_targets),
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist()
    }
    json_path = os.path.join(output_metrics_dir, f"metrics_{split_name}.json")
    with open(json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Saved metrics JSON: {json_path}")

    # 7. Save predictions dataframe for Grad-CAM
    preds_df = pd.DataFrame({
        'image_id': all_image_ids,
        'true_label_idx': all_targets,
        'true_label': [CLASSES[t] for t in all_targets],
        'pred_label_idx': all_preds,
        'pred_label': [CLASSES[p] for p in all_preds],
        'confidence': [float(probs[p]) for probs, p in zip(all_probs, all_preds)]
    })
    preds_df['correct'] = preds_df['true_label_idx'] == preds_df['pred_label_idx']
    preds_csv_path = os.path.join(output_metrics_dir, f"predictions_{split_name}.csv")
    preds_df.to_csv(preds_csv_path, index=False)
    print(f"Saved prediction records to: {preds_csv_path}")

    return metrics_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ResNet50 on ISIC 2019 dataset")
    parser.add_argument("--model_path", type=str, default="models/best_resnet50.pth")
    parser.add_argument("--eval_csv", type=str, default="data/processed/val_split.csv")
    parser.add_argument("--split_name", type=str, default="validation")
    args = parser.parse_args()

    evaluate_model(model_path=args.model_path, eval_csv=args.eval_csv, split_name=args.split_name)
