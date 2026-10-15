"""
Phase 8 & 9 (4-Class): Comprehensive Model Evaluation
Evaluates 4-class ResNet-50 on:
1. Validation Split (5,034 images)
2. Official ISIC 2019 Test Set (6,191 images mapped to NV, MEL, BCC, OTHER)

Produces:
- Accuracy, Macro F1, Weighted F1
- Per-class Precision, Recall, F1-score, Support
- Raw and Normalized Confusion Matrices (saved to results/plots/)
- JSON Metrics (saved to results/metrics/)
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
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_4class import CLASSES_4, CLASS_NAMES_4, CLASS_TO_IDX_4, LABEL_MAP
from src.train_4class import build_4class_resnet50, get_4class_transforms, FastBatchLoader4Class
from src.metrics import compute_metrics
from src.model import get_device

def evaluate_4class_split(
    model_path="models/best_resnet50_4class.pth",
    eval_csv="data/processed/val_split_4class.csv",
    split_name="validation_4class",
    output_plot_dir="results/plots",
    output_metrics_dir="results/metrics",
    batch_size=64
):
    print("=" * 60)
    print(f"4-CLASS MODEL EVALUATION: {split_name.upper()}")
    print("=" * 60)

    device = get_device()
    os.makedirs(output_plot_dir, exist_ok=True)
    os.makedirs(output_metrics_dir, exist_ok=True)

    # 1. Load Model Checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = build_4class_resnet50(pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Prepare Data
    df = pd.read_csv(eval_csv)
    print(f"Evaluating {len(df)} samples from {eval_csv}...")
    _, val_transform = get_4class_transforms(224)
    loader = FastBatchLoader4Class(df, transform=val_transform, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []
    all_probs = []
    all_image_ids = []

    with torch.no_grad():
        for images, targets, ids in loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).detach().cpu().numpy()
            preds = outputs.argmax(dim=1).detach().cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_probs.extend(probs)
            all_image_ids.extend(ids)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    # 3. Compute Metrics
    acc, macro_f1, weighted_f1, per_class_f1s = compute_metrics(all_targets, all_preds, num_classes=4)

    per_class_metrics = {}
    print("\n" + "=" * 65)
    print(f"{'Class':<8} | {'Diagnosis':<28} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8} | {'Support':<8}")
    print("-" * 80)
    for c in range(4):
        cls_code = CLASSES_4[c]
        tp = np.sum((all_preds == c) & (all_targets == c))
        fp = np.sum((all_preds == c) & (all_targets != c))
        fn = np.sum((all_preds != c) & (all_targets == c))
        support = int(np.sum(all_targets == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = per_class_f1s[c]

        per_class_metrics[cls_code] = {
            'diagnosis': CLASS_NAMES_4[cls_code],
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'support': support
        }
        print(f"{cls_code:<8} | {CLASS_NAMES_4[cls_code]:<28} | {precision:<10.4f} | {recall:<8.4f} | {f1:<8.4f} | {support:<8}")

    print("-" * 80)
    print(f"Overall 4-Class Accuracy:   {acc*100:.2f}%")
    print(f"Overall 4-Class Macro F1:   {macro_f1:.4f}")
    print(f"Overall 4-Class Weighted F1: {weighted_f1:.4f}")
    print("=" * 65)

    # 4. Confusion Matrix
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=CLASSES_4, yticklabels=CLASSES_4)
    ax1.set_title(f"4-Class Confusion Matrix (Raw Counts) - {split_name.title()}", fontweight='bold', fontsize=12)
    ax1.set_xlabel("Predicted Class", fontweight='bold')
    ax1.set_ylabel("True Ground Truth", fontweight='bold')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', ax=ax2,
                xticklabels=CLASSES_4, yticklabels=CLASSES_4)
    ax2.set_title(f"4-Class Normalized Confusion Matrix - {split_name.title()}", fontweight='bold', fontsize=12)
    ax2.set_xlabel("Predicted Class", fontweight='bold')
    ax2.set_ylabel("True Ground Truth", fontweight='bold')

    plt.tight_layout()
    cm_path = os.path.join(output_plot_dir, f"confusion_matrix_{split_name}.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix: {cm_path}")

    # 5. Save JSON summary
    summary = {
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
        json.dump(summary, f, indent=4)
    print(f"Saved metrics JSON: {json_path}")

    # 6. Save predictions DataFrame
    preds_df = pd.DataFrame({
        'image_id': all_image_ids,
        'true_label_idx': all_targets,
        'true_label': [CLASSES_4[t] for t in all_targets],
        'pred_label_idx': all_preds,
        'pred_label': [CLASSES_4[p] for p in all_preds],
        'confidence': [float(probs[p]) for probs, p in zip(all_probs, all_preds)]
    })
    preds_df['correct'] = preds_df['true_label_idx'] == preds_df['pred_label_idx']
    preds_csv = os.path.join(output_metrics_dir, f"predictions_{split_name}.csv")
    preds_df.to_csv(preds_csv, index=False)
    print(f"Saved prediction records to: {preds_csv}")

    return summary

def prepare_and_evaluate_4class_test(
    test_gt_csv="ISIC_2019_Test_GroundTruth.csv",
    test_img_dir="data/raw/test/ISIC_2019_Test_Input",
    output_processed_csv="data/processed/test_split_4class.csv",
    model_path="models/best_resnet50_4class.pth"
):
    print("\n" + "=" * 60)
    print("PREPARING OFFICIAL ISIC 2019 TEST SET (4-CLASS MAPPING)")
    print("=" * 60)

    df_test_gt = pd.read_csv(test_gt_csv)
    original_classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    test_records = []

    for _, row in df_test_gt.iterrows():
        img_id = row['image']
        img_path = os.path.join(test_img_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue

        scores = [row[c] for c in original_classes if c in row]
        if max(scores) == 1.0:
            assigned_8class = original_classes[int(np.argmax(scores))]
            assigned_4class = LABEL_MAP[assigned_8class]
            test_records.append({
                'image_id': img_id,
                'image_path': img_path,
                'original_label': assigned_8class,
                'label_4class': assigned_4class,
                'label_idx_4class': CLASS_TO_IDX_4[assigned_4class]
            })

    df_test_4class = pd.DataFrame(test_records)
    df_test_4class.to_csv(output_processed_csv, index=False)
    print(f"Prepared 4-class test set: {len(df_test_4class)} images -> {output_processed_csv}")

    # Evaluate
    test_summary = evaluate_4class_split(
        model_path=model_path,
        eval_csv=output_processed_csv,
        split_name="test_4class",
        batch_size=64
    )
    return test_summary

if __name__ == "__main__":
    # 1. Validation evaluation
    evaluate_4class_split(
        model_path="models/best_resnet50_4class.pth",
        eval_csv="data/processed/val_split_4class.csv",
        split_name="validation_4class"
    )

    # 2. Official Test evaluation
    prepare_and_evaluate_4class_test()
