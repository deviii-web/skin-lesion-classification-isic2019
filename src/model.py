"""
Phase 5 & 6: Pretrained ResNet-50 Model and Class-Imbalance Handling
Safe initialization compatible with Python 3.13 / Apple Silicon macOS.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
NUM_CLASSES = len(CLASSES)

def build_resnet50(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=False):
    """
    Constructs ResNet-50.
    Uses safe weights initialization to avoid Python 3.13 / torchvision hook segfaults.
    """
    model = models.resnet50(weights=None)

    if pretrained:
        weights_file = "models/resnet50_imagenet.pth"
        if os.path.exists(weights_file):
            state = torch.load(weights_file, weights_only=True)
            # Exclude final fc layer to allow custom number of classes
            filtered_state = {k: v for k, v in state.items() if not k.startswith("fc.")}
            model.load_state_dict(filtered_state, strict=False)
        else:
            try:
                w = models.ResNet50_Weights.DEFAULT
                state = w.get_state_dict(progress=False)
                os.makedirs("models", exist_ok=True)
                torch.save(state, weights_file)
                filtered_state = {k: v for k, v in state.items() if not k.startswith("fc.")}
                model.load_state_dict(filtered_state, strict=False)
            except Exception as e:
                print(f"Warning: could not load pretrained weights: {e}")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classifier
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

def compute_class_weights(train_csv_or_df):
    """
    Calculates balanced class weights strictly from the training split.
    """
    if isinstance(train_csv_or_df, str):
        df = pd.read_csv(train_csv_or_df)
    else:
        df = train_csv_or_df

    total_samples = len(df)
    class_counts = df['label_idx'].value_counts().sort_index()

    weights = []
    for idx in range(NUM_CLASSES):
        count = class_counts.get(idx, 0)
        w = total_samples / (NUM_CLASSES * count) if count > 0 else 1.0
        weights.append(w)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    weights_tensor = weights_tensor / weights_tensor.mean()
    return weights_tensor

def get_loss_function(train_csv="data/processed/train_split.csv", device="cpu"):
    weights = compute_class_weights(train_csv).to(device)
    return nn.CrossEntropyLoss(weight=weights)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")
