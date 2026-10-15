"""
Phase 1 (4-Class): Dataset Preparation and Lesion-Safe Stratified Split
Maps original 8 ISIC classes into 4 focused categories:
  NV   -> NV    (0)
  MEL  -> MEL   (1)
  BCC  -> BCC   (2)
  AK, BKL, DF, VASC, SCC -> OTHER (3)

Preserves ALL 25,331 images.
Creates lesion-safe stratified 80/20 train/validation splits:
  - data/processed/train_split_4class.csv
  - data/processed/val_split_4class.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold

CLASSES_4 = ['NV', 'MEL', 'BCC', 'OTHER']
CLASS_NAMES_4 = {
    'NV': 'Melanocytic Nevus',
    'MEL': 'Melanoma',
    'BCC': 'Basal Cell Carcinoma',
    'OTHER': 'Other Lesions (AK, BKL, DF, VASC, SCC)'
}
CLASS_TO_IDX_4 = {cls: idx for idx, cls in enumerate(CLASSES_4)}

LABEL_MAP = {
    'NV': 'NV',
    'MEL': 'MEL',
    'BCC': 'BCC',
    'AK': 'OTHER',
    'BKL': 'OTHER',
    'DF': 'OTHER',
    'VASC': 'OTHER',
    'SCC': 'OTHER'
}

def process_4class_dataset(
    input_csv="data/processed/train_validated.csv",
    output_train_csv="data/processed/train_split_4class.csv",
    output_val_csv="data/processed/val_split_4class.csv",
    output_eda_plot="results/plots/eda_4class_distribution.png"
):
    print("=" * 60)
    print("4-CLASS DATASET PREPARATION & LESION-SAFE SPLIT")
    print("=" * 60)

    df = pd.read_csv(input_csv)
    print(f"Loaded validated dataset: {len(df)} images")

    # 1. Apply 4-class mapping
    df['original_label'] = df['label']
    df['label_4class'] = df['original_label'].map(LABEL_MAP)
    df['label_idx_4class'] = df['label_4class'].map(CLASS_TO_IDX_4)

    # 2. Verify counts
    print("\nClass Counts Verification:")
    counts_8class = df['original_label'].value_counts()
    other_expected = (
        counts_8class.get('AK', 0) +
        counts_8class.get('BKL', 0) +
        counts_8class.get('DF', 0) +
        counts_8class.get('VASC', 0) +
        counts_8class.get('SCC', 0)
    )
    counts_4class = df['label_4class'].value_counts()
    print(f"  - NV:    {counts_4class['NV']} (expected: {counts_8class['NV']})")
    print(f"  - MEL:   {counts_4class['MEL']} (expected: {counts_8class['MEL']})")
    print(f"  - BCC:   {counts_4class['BCC']} (expected: {counts_8class['BCC']})")
    print(f"  - OTHER: {counts_4class['OTHER']} (expected sum AK+BKL+DF+VASC+SCC: {other_expected})")

    assert counts_4class['OTHER'] == other_expected, "Mismatch in OTHER class aggregation count!"
    print("\n[OK] 4-class aggregation count verified!")

    # 3. Generate 4-class EDA distribution plot
    os.makedirs(os.path.dirname(output_eda_plot), exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(9, 6))

    class_order = ['NV', 'MEL', 'BCC', 'OTHER']
    plot_counts = [counts_4class[c] for c in class_order]
    palette = sns.color_palette("Set2", 4)
    bars = plt.bar([f"{c}\n({CLASS_NAMES_4[c]})" for c in class_order], plot_counts, color=palette)

    plt.title("ISIC 2019 4-Class Diagnostic Distribution", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Number of Images", fontsize=12)
    plt.ylim(0, max(plot_counts) * 1.15)

    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        plt.text(bar.get_x() + bar.get_width()/2., h + 250,
                 f"{h:,}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_eda_plot, dpi=300)
    plt.close()
    print(f"Saved 4-class EDA plot to: {output_eda_plot}")

    # 4. Create Lesion-Safe Stratified Split (80% train / 20% val)
    print("\nCreating 80/20 StratifiedGroupKFold split grouped by lesion_id...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(sgkf.split(df, df['label_idx_4class'], groups=df['lesion_id']))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)

    # 5. Verify Zero Lesion Overlap
    train_lesions = set(train_df['lesion_id'])
    val_lesions = set(val_df['lesion_id'])
    overlap = train_lesions.intersection(val_lesions)
    print(f"Lesion ID Overlap between Train and Val: {len(overlap)} (MUST BE 0)")
    assert len(overlap) == 0, "Data leakage detected: overlapping lesion IDs!"

    # Save to disk
    train_df.to_csv(output_train_csv, index=False)
    val_df.to_csv(output_val_csv, index=False)
    print(f"\nSaved 4-Class Train Split ({len(train_df)} samples, {len(train_df)/len(df)*100:.2f}%) -> {output_train_csv}")
    print(f"Saved 4-Class Val Split   ({len(val_df)} samples, {len(val_df)/len(df)*100:.2f}%) -> {output_val_csv}")

    # Compare class distributions
    print("\n4-Class Distribution Comparison:")
    print(f"{'Class':<8} | {'Train Count':<12} | {'Train %':<8} | {'Val Count':<10} | {'Val %':<8}")
    print("-" * 55)
    for cls in class_order:
        tr_cnt = (train_df['label_4class'] == cls).sum()
        tr_pct = tr_cnt / len(train_df) * 100
        val_cnt = (val_df['label_4class'] == cls).sum()
        val_pct = val_cnt / len(val_df) * 100
        print(f"{cls:<8} | {tr_cnt:<12} | {tr_pct:<7.2f}% | {val_cnt:<10} | {val_pct:<7.2f}%")
    print("=" * 60)

    return train_df, val_df

if __name__ == "__main__":
    process_4class_dataset()
