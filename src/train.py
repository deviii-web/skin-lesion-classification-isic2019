"""
Phase 7: Production Training Script for ISIC 2019
Fully stabilized with FastBatchLoader, native metrics, and checkpointing.
"""

import os
import sys
import time
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import FastBatchLoader, get_transforms, CLASSES, NUM_CLASSES
from src.model import build_resnet50, compute_class_weights, get_device
from src.metrics import compute_metrics

def run_training(
    train_csv="data/processed/train_split.csv",
    val_csv="data/processed/val_split.csv",
    epochs=5,
    batch_size=32,
    lr=3e-4,
    save_model_path="models/best_resnet50.pth",
    history_json_path="results/metrics/training_history.json",
    sample_limit=None
):
    print("=" * 60)
    print("PHASE 7: Model Training (ResNet-50 on ISIC 2019)")
    print("=" * 60)

    device = get_device()
    print(f"Using accelerator: {device}")

    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_json_path), exist_ok=True)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    if sample_limit:
        print(f"Subsampling to {sample_limit} rows for training...")
        train_df = train_df.sample(min(sample_limit, len(train_df)), random_state=42).reset_index(drop=True)
        val_df = val_df.sample(min(sample_limit // 4, len(val_df)), random_state=42).reset_index(drop=True)

    print(f"Training dataset: {len(train_df)} samples | Validation dataset: {len(val_df)} samples")

    train_transform, val_transform = get_transforms(224)
    train_loader = FastBatchLoader(train_df, transform=train_transform, batch_size=batch_size, shuffle=True)
    val_loader = FastBatchLoader(val_df, transform=val_transform, batch_size=batch_size, shuffle=False)

    model = build_resnet50(num_classes=NUM_CLASSES, pretrained=True).to(device)
    class_weights = compute_class_weights(train_df).to(device)

    print("\nComputed Normalized Class Weights from Training Split:")
    for idx, (cls, w) in enumerate(zip(CLASSES, class_weights.cpu().numpy())):
        print(f"  Class {idx} ({cls:<5}): {w:.4f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_macro_f1 = 0.0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [], 'val_weighted_f1': []
    }

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch [{epoch}/{epochs}] ---")
        t0 = time.time()

        # Training Loop
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

            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == len(train_loader):
                print(f"  Batch [{b_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        train_loss = train_loss_total / len(train_df)
        train_acc, train_f1, _, _ = compute_metrics(train_targets, train_preds, num_classes=NUM_CLASSES)

        # Validation Loop
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
        val_acc, val_macro_f1, val_weighted_f1, _ = compute_metrics(val_targets, val_preds, num_classes=NUM_CLASSES)
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

        # Save checkpoint
        if val_macro_f1 >= best_macro_f1 or epoch == 1:
            best_macro_f1 = val_macro_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_macro_f1': val_macro_f1,
                'val_acc': val_acc,
                'classes': CLASSES
            }, save_model_path)
            print(f"-> Checkpoint Saved: Best Val Macro F1: {best_macro_f1:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation Macro F1: {best_macro_f1:.4f}")

    # Save metrics JSON
    with open(history_json_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Saved training history to {history_json_path}")

    # Plot training curves
    epochs_range = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss')
    ax1.plot(epochs_range, history['val_loss'], 'r-o', label='Val Loss')
    ax1.set_title('Cross-Entropy Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, history['train_acc'], 'b-o', label='Train Accuracy')
    ax2.plot(epochs_range, history['val_acc'], 'r-o', label='Val Accuracy')
    ax2.set_title('Accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(epochs_range, history['train_f1'], 'b-o', label='Train Macro F1')
    ax3.plot(epochs_range, history['val_f1'], 'r-o', label='Val Macro F1')
    ax3.set_title('Macro F1 Score', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Macro F1')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    curve_path = "results/plots/training_curves.png"
    os.makedirs(os.path.dirname(curve_path), exist_ok=True)
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {curve_path}")

    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet50 on ISIC 2019")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--sample_limit", type=int, default=1500, help="Number of samples to train")
    args = parser.parse_args()

    run_training(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, sample_limit=args.sample_limit)
