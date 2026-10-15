"""
ISIC 2019 Dataset Pipeline with FastBatchLoader
"""

import os
import math
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

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
NUM_CLASSES = len(CLASSES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, val_transform

class FastBatchLoader:
    """
    In-process, single-threaded batch generator.
    Guarantees zero multi-processing fork crashes on macOS / Python 3.13.
    """
    def __init__(self, df_or_csv, transform=None, batch_size=32, shuffle=True):
        if isinstance(df_or_csv, str):
            self.df = pd.read_csv(df_or_csv)
        else:
            self.df = df_or_csv.copy().reset_index(drop=True)

        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    @property
    def dataset_len(self):
        return len(self.df)

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
                        batch_labels.append(int(row['label_idx']))
                        batch_ids.append(row['image_id'])
                except Exception as e:
                    print(f"Warning: error loading {row['image_path']}: {e}")

            if batch_tensors:
                images = torch.stack(batch_tensors)
                targets = torch.tensor(batch_labels, dtype=torch.long)
                yield images, targets, batch_ids

def get_dataloaders(train_csv="data/processed/train_split.csv",
                    val_csv="data/processed/val_split.csv",
                    batch_size=32,
                    img_size=224):
    train_transform, val_transform = get_transforms(img_size=img_size)
    train_loader = FastBatchLoader(train_csv, transform=train_transform, batch_size=batch_size, shuffle=True)
    val_loader = FastBatchLoader(val_csv, transform=val_transform, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
