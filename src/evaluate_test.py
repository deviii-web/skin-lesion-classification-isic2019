"""
Phase 9: Official ISIC 2019 Test Set Preparation & Evaluation
Loads official test ground truth and images, maps classes, and runs final independent evaluation.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import CLASSES, CLASS_NAMES
from src.evaluate import evaluate_model

def prepare_and_evaluate_test(
    test_gt_csv="ISIC_2019_Test_GroundTruth.csv",
    test_img_dir="data/raw/test/ISIC_2019_Test_Input",
    output_processed_test_csv="data/processed/test_validated.csv",
    model_path="models/best_resnet50.pth"
):
    print("=" * 60)
    print("PHASE 9: Official ISIC 2019 Test Evaluation")
    print("=" * 60)

    # 1. Load test ground truth
    df_test_gt = pd.read_csv(test_gt_csv)
    print(f"Loaded test ground truth CSV: {len(df_test_gt)} images")

    # Filter out UNK/out-of-distribution classes if only evaluating the 8 diagnostic targets
    # (ISIC 2019 test set includes an UNK class with outlier images)
    test_records = []
    unk_count = 0
    missing_img = 0

    label_cols = [c for c in CLASSES if c in df_test_gt.columns]

    for _, row in df_test_gt.iterrows():
        img_id = row['image']
        img_path = os.path.join(test_img_dir, f"{img_id}.jpg")

        if not os.path.exists(img_path):
            missing_img += 1
            continue

        # Check if the row belongs to one of the 8 trained classes
        scores = [row[c] for c in label_cols]
        if max(scores) == 1.0:
            assigned_label = label_cols[int(np.argmax(scores))]
            test_records.append({
                'image_id': img_id,
                'image_path': img_path,
                'label': assigned_label,
                'label_idx': CLASSES.index(assigned_label)
            })
        else:
            unk_count += 1

    df_test_processed = pd.DataFrame(test_records)
    print(f"\nTest Dataset Summary:")
    print(f"  - Verified 8-class test images: {len(df_test_processed)}")
    print(f"  - UNK / Outlier images (excluded from 8-class metric): {unk_count}")
    print(f"  - Missing test image files: {missing_img}")

    os.makedirs(os.path.dirname(output_processed_test_csv), exist_ok=True)
    df_test_processed.to_csv(output_processed_test_csv, index=False)
    print(f"Saved processed test set to: {output_processed_test_csv}")

    # 2. Run Evaluation
    test_metrics = evaluate_model(
        model_path=model_path,
        eval_csv=output_processed_test_csv,
        split_name="test",
        batch_size=64
    )
    return test_metrics

if __name__ == "__main__":
    prepare_and_evaluate_test()
