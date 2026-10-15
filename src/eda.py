"""
Phase 2: Exploratory Data Analysis (EDA)
Performs basic, focused EDA on ISIC 2019 dataset:
- Class distribution bar chart
- Sample images grid across all 8 classes
- Image dimensions and aspect ratio analysis
- Missing data / integrity summary
Saves all visual figures to results/plots/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
CLASS_NAMES = {
    'MEL': 'Melanoma (MEL)',
    'NV': 'Melanocytic Nevus (NV)',
    'BCC': 'Basal Cell Carcinoma (BCC)',
    'AK': 'Actinic Keratosis (AK)',
    'BKL': 'Benign Keratosis (BKL)',
    'DF': 'Dermatofibroma (DF)',
    'VASC': 'Vascular Lesion (VASC)',
    'SCC': 'Squamous Cell Carcinoma (SCC)'
}

def run_eda(csv_path="data/processed/train_validated.csv", output_dir="results/plots"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    print("=" * 60)
    print("PHASE 2: Exploratory Data Analysis (EDA)")
    print("=" * 60)
    print(f"Dataset shape: {df.shape}")

    # Set style
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Class Distribution Plot
    print("\n1. Generating Class Distribution Plot...")
    plt.figure(figsize=(10, 6))
    counts = [df[df['label'] == c].shape[0] for c in CLASSES]
    palette = sns.color_palette("deep", len(CLASSES))

    bars = plt.bar([CLASS_NAMES[c] for c in CLASSES], counts, color=palette)
    plt.title("ISIC 2019 Diagnostic Class Distribution", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Lesion Diagnosis", fontsize=12, labelpad=10)
    plt.ylabel("Number of Images", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=10)

    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        plt.text(bar.get_x() + bar.get_width()/2., height + 150,
                 f'{height:,}\n({pct:.1f}%)',
                 ha='center', va='bottom', fontsize=9)

    plt.ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "eda_class_distribution.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"Saved: {dist_path}")

    # 2. Sample Images Grid (2 samples per class -> 2x8 grid)
    print("\n2. Generating Sample Images Grid...")
    fig, axes = plt.subplots(2, len(CLASSES), figsize=(18, 5.5))

    for col_idx, cls in enumerate(CLASSES):
        cls_df = df[df['label'] == cls]
        # Pick 2 representative samples
        samples = cls_df.sample(n=min(2, len(cls_df)), random_state=42)
        for row_idx, (_, row) in enumerate(samples.iterrows()):
            ax = axes[row_idx, col_idx]
            img = Image.open(row['image_path']).convert('RGB')
            ax.imshow(img)
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(cls, fontsize=12, fontweight='bold')

    plt.suptitle("ISIC 2019 Representative Dermoscopic Images by Diagnostic Class", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    samples_path = os.path.join(output_dir, "eda_sample_images.png")
    plt.savefig(samples_path, dpi=300)
    plt.close()
    print(f"Saved: {samples_path}")

    # 3. Image Dimensions & Aspect Ratio Analysis (Sample 500 images for speed)
    print("\n3. Analyzing Image Dimensions...")
    sample_sub = df.sample(n=min(1000, len(df)), random_state=42)
    widths, heights, aspect_ratios = [], [], []
    for _, row in sample_sub.iterrows():
        with Image.open(row['image_path']) as img:
            w, h = img.size
            widths.append(w)
            height.append(h) if False else None
            heights.append(h)
            aspect_ratios.append(w / h)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(widths, heights, alpha=0.5, color='teal', edgecolors='none', s=25)
    ax1.set_title("Image Resolution Distribution (Sample = 1000)", fontweight='bold')
    ax1.set_xlabel("Width (px)")
    ax1.set_ylabel("Height (px)")

    sns.histplot(aspect_ratios, bins=25, kde=True, ax=ax2, color='coral')
    ax2.set_title("Aspect Ratio (Width / Height) Distribution", fontweight='bold')
    ax2.set_xlabel("Aspect Ratio")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    dim_path = os.path.join(output_dir, "eda_image_dimensions.png")
    plt.savefig(dim_path, dpi=300)
    plt.close()
    print(f"Saved: {dim_path}")

    # Summary Output
    print("\nEDA Findings Summary:")
    print(f"- Class Imbalance: NV comprises {df['label'].value_counts()['NV'] / len(df) * 100:.2f}% of data, while DF ({df['label'].value_counts()['DF'] / len(df) * 100:.2f}%) and VASC ({df['label'].value_counts()['VASC'] / len(df) * 100:.2f}%) are heavily minority classes.")
    print(f"- Image Dimensions: Typical image sizes range from {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)} px.")
    print(f"- Aspect ratios cluster tightly around {np.mean(aspect_ratios):.2f}.")
    print("=" * 60)

if __name__ == "__main__":
    run_eda()
