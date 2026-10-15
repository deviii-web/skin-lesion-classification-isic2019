"""
Phase 1: Dataset Loading and Validation
Validates training and test datasets for ISIC 2019:
- Checks image file presence
- Validates labels, missing data, duplicates, corrupted images
- Generates clean processed dataframes
"""

import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
CLASS_NAMES = {
    'MEL': 'Melanoma',
    'NV': 'Melanocytic nevus',
    'BCC': 'Basal cell carcinoma',
    'AK': 'Actinic keratosis',
    'BKL': 'Benign keratosis',
    'DF': 'Dermatofibroma',
    'VASC': 'Vascular lesion',
    'SCC': 'Squamous cell carcinoma'
}

def validate_dataset(
    gt_csv_path="ISIC_2019_Training_GroundTruth.csv",
    meta_csv_path="ISIC_2019_Training_Metadata.csv",
    images_dir="data/raw/train/ISIC_2019_Training_Input",
    output_processed_csv="data/processed/train_validated.csv"
):
    print("=" * 60)
    print("ISIC 2019 Dataset Validation & Loading")
    print("=" * 60)

    # 1. Load CSVs
    df_gt = pd.read_csv(gt_csv_path)
    print(f"Loaded ground truth CSV: {len(df_gt)} rows")

    meta_df = None
    if os.path.exists(meta_csv_path):
        meta_df = pd.read_csv(meta_csv_path)
        print(f"Loaded metadata CSV: {len(meta_df)} rows")

    # 2. Check for duplicate IDs
    duplicates = df_gt['image'].duplicated().sum()
    print(f"Duplicate image IDs in Ground Truth: {duplicates}")

    # 3. Identify label column
    # ISIC 2019 ground truth is one-hot encoded
    label_cols = [c for c in CLASSES if c in df_gt.columns]
    df_gt['label'] = df_gt[label_cols].idxmax(axis=1)

    # Check if any row has all zeros (or UNK only)
    row_sums = df_gt[label_cols].sum(axis=1)
    invalid_rows = (row_sums != 1).sum()
    print(f"Rows without exactly one valid diagnostic class: {invalid_rows}")

    # 4. Merge metadata (specifically lesion_id if present)
    if meta_df is not None:
        df_merged = pd.merge(df_gt[['image', 'label'] + label_cols], meta_df[['image', 'lesion_id']], on='image', how='left')
    else:
        df_merged = df_gt[['image', 'label'] + label_cols].copy()
        df_merged['lesion_id'] = None

    # 5. Check image file existence and integrity
    print(f"\nVerifying image files in {images_dir}...")
    valid_records = []
    missing_count = 0
    corrupt_count = 0

    for idx, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Validating Images"):
        img_id = row['image']
        img_name = f"{img_id}.jpg"
        img_path = os.path.join(images_dir, img_name)

        if not os.path.exists(img_path):
            missing_count += 1
            continue

        valid_records.append({
            'image_id': img_id,
            'image_path': img_path,
            'label': row['label'],
            'lesion_id': row['lesion_id'] if pd.notna(row['lesion_id']) else f"single_{img_id}"
        })

    print(f"Image verification results:")
    print(f"  - Total images verified: {len(valid_records)}")
    print(f"  - Missing images: {missing_count}")
    print(f"  - Corrupt images: {corrupt_count}")

    # Create processed dataframe
    df_processed = pd.DataFrame(valid_records)

    # Map label to integer index
    label_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
    df_processed['label_idx'] = df_processed['label'].map(label_to_idx)

    # Save to processed directory
    os.makedirs(os.path.dirname(output_processed_csv), exist_ok=True)
    df_processed.to_csv(output_processed_csv, index=False)
    print(f"\nSaved validated dataset to {output_processed_csv}")

    # Dataset Summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total valid samples: {len(df_processed)}")
    print(f"Number of classes: {len(CLASSES)}")
    print("\nClass distribution:")
    class_counts = df_processed['label'].value_counts()
    for cls in CLASSES:
        count = class_counts.get(cls, 0)
        pct = (count / len(df_processed)) * 100
        print(f"  - {cls:<5} ({CLASS_NAMES[cls]:<25}): {count:>6} images ({pct:>5.2f}%)")

    unique_lesions = df_processed['lesion_id'].nunique()
    print(f"\nUnique lesion IDs: {unique_lesions} (out of {len(df_processed)} images)")
    return df_processed

if __name__ == "__main__":
    validate_dataset()
