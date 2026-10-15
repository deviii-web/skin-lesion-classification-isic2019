"""
4-Class Model Training & Fine-Tuning Pipeline
Features:
- Pretrained ResNet-50 backbone with 4-class classification head (NV=0, MEL=1, BCC=2, OTHER=3)
- Balanced class weights computed strictly from the 4-class training split
- In-process FastBatchLoader for memory stability
- Learning rate warmup / step decay scheduler
- Native metric calculation (Accuracy, Macro F1, Weighted F1, Loss)
- Best checkpoint saved to models/best_resnet50_4class.pth
- Training curves saved to results/plots/training_curves_4class.png
"""

import os
import sys
import time
import json
import math
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_4class import CLASSES_4, CLASS_NAMES_4, CLASS_TO_IDX_4
from src.metrics import compute_metrics
from src.model import get_device

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_4class_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, val_transform

class FastBatchLoader4Class:
    def __init__(self, df, transform=None, batch_size=32, shuffle=True):
        self.df = df.copy().reset_index(drop=True)
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    def __iter__(self):
        idxs = self.indices.copy()
        if self.shuffle:
            np.random.shuffle(idxs)

        for start_idx in range(0, len(idxs), self.batch_size):
            batch_idxs = idxs[start_idx:start_idx + self.batch_size]
            batch_tensors = []
            batch_labels = []
            batch_ids = []

            for idx in batch_idxs:
                row = self.df.iloc[idx]
                try:
                    with Image.open(row['image_path']) as img:
                        img_rgb = img.convert('RGB')
                        if self.transform:
                            tensor = self.transform(img_rgb)
                        else:
                            tensor = transforms.ToTensor()(img_rgb)
                        batch_tensors.append(tensor)
                        batch_labels.append(int(row['label_idx_4class']))
                        batch_ids.append(row['image_id'])
                except Exception as e:
                    print(f"Warning: error loading {row['image_path']}: {e}")

            if batch_tensors:
                images = torch.stack(batch_tensors)
                targets = torch.tensor(batch_labels, dtype=torch.long)
                yield images, targets, batch_ids

def build_4class_resnet50(pretrained=True):
    model = models.resnet50(weights=None)
    if pretrained:
        weights_file = "models/resnet50_imagenet.pth"
        if os.path.exists(weights_file):
            state = torch.load(weights_file, weights_only=True)
            filtered_state = {k: v for k, v in state.items() if not k.startswith("fc.")}
            model.load_state_dict(filtered_state, strict=False)
        else:
            w = models.ResNet50_Weights.DEFAULT
            state = w.get_state_dict(progress=False)
            filtered_state = {k: v for k, v in state.items() if not k.startswith("fc.")}
            model.load_state_dict(filtered_state, strict=False)

    num_ftrs = model.fc.in_features
    # 4 classes: NV, MEL, BCC, OTHER
    model.fc = nn.Linear(num_ftrs, 4)
    return model

def compute_4class_weights(train_df):
    total = len(train_df)
    counts = train_df['label_idx_4class'].value_counts().sort_index()
    weights = []
    for idx in range(4):
        c = counts.get(idx, 0)
        # Smoothed inverse frequency
        w = total / (4.0 * c) if c > 0 else 1.0
        weights.append(w)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalize weights around mean 1.0
    weights_tensor = weights_tensor / weights_tensor.mean()
    return weights_tensor

def train_4class_model(
    train_csv="data/processed/train_split_4class.csv",
    val_csv="data/processed/val_split_4class.csv",
    epochs=5,
    batch_size=32,
    lr=2e-4,
    save_model_path="models/best_resnet50_4class.pth",
    history_json_path="results/metrics/training_history_4class.json",
    sample_limit=2500
):
    print("=" * 60)
    print("4-CLASS RESNET-50 MODEL TRAINING & FINE-TUNING")
    print("=" * 60)

    device = get_device()
    print(f"Using compute accelerator: {device}")

    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_json_path), exist_ok=True)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    if sample_limit:
        print(f"Subsampling to {sample_limit} rows for optimized transfer learning...")
        train_df = train_df.sample(min(sample_limit, len(train_df)), random_state=42).reset_index(drop=True)
        val_df = val_df.sample(min(sample_limit // 4, len(val_df)), random_state=42).reset_index(drop=True)

    print(f"Training dataset: {len(train_df)} samples | Validation dataset: {len(val_df)} samples")

    train_transform, val_transform = get_4class_transforms(224)
    train_loader = FastBatchLoader4Class(train_df, transform=train_transform, batch_size=batch_size, shuffle=True)
    val_loader = FastBatchLoader4Class(val_df, transform=val_transform, batch_size=batch_size, shuffle=False)

    model = build_4class_resnet50(pretrained=True).to(device)
    class_weights = compute_4class_weights(train_df).to(device)

    print("\nComputed 4-Class Normalized Weights from Training Set:")
    for idx, (cls, w) in enumerate(zip(CLASSES_4, class_weights.cpu().numpy())):
        print(f"  Class {idx} ({cls:<6} - {CLASS_NAMES_4[cls]:<25}): {w:.4f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Step learning rate scheduler for clean convergence
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_acc = 0.0
    best_macro_f1 = 0.0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [], 'val_weighted_f1': []
    }

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch [{epoch}/{epochs}] (lr: {optimizer.param_groups[0]['lr']:.6f}) ---")
        t0 = time.time()

        # Training Phase
        model.train()
        train_loss_total = 0.0
        train_preds, train_targets = [], []

        for b_idx, (images, targets, _) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(targets.detach().cpu().numpy())

            if (b_idx + 1) % 15 == 0 or (b_idx + 1) == len(train_loader):
                print(f"  Batch [{b_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        scheduler.step()

        train_loss = train_loss_total / len(train_df)
        train_acc, train_f1, _, _ = compute_metrics(train_targets, train_preds, num_classes=4)

        # Validation Phase
        model.eval()
        val_loss_total = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for images, targets, _ in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss_total += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1).detach().cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(targets.detach().cpu().numpy())

        val_loss = val_loss_total / len(val_df)
        val_acc, val_macro_f1, val_weighted_f1, _ = compute_metrics(val_targets, val_preds, num_classes=4)
        epoch_time = time.time() - t0

        print(f"Epoch {epoch:02d} ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% Macro F1: {val_macro_f1:.4f} Weighted F1: {val_weighted_f1:.4f}")

        # Record metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_macro_f1)
        history['val_weighted_f1'].append(val_weighted_f1)

        # Save checkpoint based on validation accuracy & Macro F1
        if val_acc > best_acc or epoch == 1:
            best_acc = val_acc
            best_macro_f1 = val_macro_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_macro_f1': val_macro_f1,
                'classes': CLASSES_4,
                'class_names': CLASS_NAMES_4
            }, save_model_path)
            print(f"-> Checkpoint Saved: Best Val Acc: {best_acc*100:.2f}% | Best Macro F1: {best_macro_f1:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    print(f"Best 4-Class Validation Accuracy: {best_acc*100:.2f}% | Macro F1: {best_macro_f1:.4f}")

    # Save metrics JSON
    with open(history_json_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Saved training history to {history_json_path}")

    # Plot training curves
    epochs_range = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss')
    ax1.plot(epochs_range, history['val_loss'], 'r-o', label='Val Loss')
    ax1.set_title('Cross-Entropy Loss (4-Class)', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, history['train_acc'], 'b-o', label='Train Accuracy')
    ax2.plot(epochs_range, history['val_acc'], 'r-o', label='Val Accuracy')
    ax2.set_title('Accuracy (4-Class)', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(epochs_range, history['train_f1'], 'b-o', label='Train Macro F1')
    ax3.plot(epochs_range, history['val_f1'], 'r-o', label='Val Macro F1')
    ax3.set_title('Macro F1 Score (4-Class)', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Macro F1')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    curve_path = "results/plots/training_curves_4class.png"
    os.makedirs(os.path.dirname(curve_path), exist_ok=True)
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {curve_path}")

    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 4-Class ResNet50 on ISIC 2019")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--sample_limit", type=int, default=2500, help="Number of samples for efficient transfer learning")
    args = parser.parse_args()

    train_4class_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, sample_limit=args.sample_limit)
