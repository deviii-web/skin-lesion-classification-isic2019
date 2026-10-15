"""
Phase 3: Train / Validation Split
Creates a reproducible, stratified 80% train / 20% validation split.
Checks lesion_id grouping to prevent data leakage (images from the same patient/lesion
stay strictly within either the train or the validation partition).
Saves train_split.csv and val_split.csv to data/processed/.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

def create_splits(
    input_csv="data/processed/train_validated.csv",
    output_train_csv="data/processed/train_split.csv",
    output_val_csv="data/processed/val_split.csv",
    val_ratio=0.20,
    random_state=42
):
    print("=" * 60)
    print("PHASE 3: Train / Validation Split")
    print("=" * 60)

    df = pd.read_csv(input_csv)
    print(f"Loaded dataset: {len(df)} samples")

    # Check lesion_id availability
    has_lesions = 'lesion_id' in df.columns and df['lesion_id'].notna().any()

    if has_lesions:
        print(f"Grouping by lesion_id ({df['lesion_id'].nunique()} unique lesions) to avoid patient leakage...")
        # Use StratifiedGroupKFold with 5 splits (which gives exactly 80% train / 20% val)
        sgkf = StratifiedGroupKFold(n_splits=int(1/val_ratio), shuffle=True, random_state=random_state)
        train_idx, val_idx = next(sgkf.split(df, df['label_idx'], groups=df['lesion_id']))
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        val_df = df.iloc[val_idx].copy().reset_index(drop=True)
    else:
        print("Using standard StratifiedShuffleSplit (no lesion grouping available)...")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
        train_idx, val_idx = next(sss.split(df, df['label_idx']))
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        val_df = df.iloc[val_idx].copy().reset_index(drop=True)

    # Verify no lesion overlap
    train_lesions = set(train_df['lesion_id'])
    val_lesions = set(val_df['lesion_id'])
    overlap = train_lesions.intersection(val_lesions)
    print(f"Lesion ID overlap between Train and Val: {len(overlap)} (MUST BE 0)")
    assert len(overlap) == 0, "Data leakage detected: overlapping lesion IDs!"

    # Save splits
    train_df.to_csv(output_train_csv, index=False)
    val_df.to_csv(output_val_csv, index=False)
    print(f"\nSaved Train Split ({len(train_df)} images, {len(train_df)/len(df)*100:.2f}%) -> {output_train_csv}")
    print(f"Saved Val Split   ({len(val_df)} images, {len(val_df)/len(df)*100:.2f}%) -> {output_val_csv}")

    # Compare class distributions
    print("\nClass Distribution Comparison:")
    print(f"{'Class':<6} | {'Train Count':<12} | {'Train %':<8} | {'Val Count':<10} | {'Val %':<8}")
    print("-" * 55)
    for cls in CLASSES:
        tr_cnt = (train_df['label'] == cls).sum()
        tr_pct = tr_cnt / len(train_df) * 100
        val_cnt = (val_df['label'] == cls).sum()
        val_pct = val_cnt / len(val_df) * 100
        print(f"{cls:<6} | {tr_cnt:<12} | {tr_pct:<7.2f}% | {val_cnt:<10} | {val_pct:<7.2f}%")
    print("=" * 60)

    return train_df, val_df

if __name__ == "__main__":
    create_splits()
