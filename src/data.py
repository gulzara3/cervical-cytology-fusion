"""Dataset, augmentation transforms, and the stratified leakage-free split.

Augmentation is applied ONLY to the training transform; validation and test use
resize + ImageNet normalisation, so no augmented variant of a training image
reaches the held-out sets. The split is stratified and seeded (random_state=42),
performed at the individual-cell level BEFORE augmentation. The exact held-out
test indices used in the paper are provided in results/test_idx.npy.
"""
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

from .config import config


class MultiModalDataset(Dataset):
    def __init__(self, image_paths, clinical, genomic, labels, transform=None, modality_dropout=0.0, training=True):
        self.image_paths = image_paths
        self.clinical = clinical
        self.genomic = genomic
        self.labels = labels
        self.transform = transform
        self.modality_dropout = modality_dropout
        self.training = training

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        try:
            img = cv2.imread(self.image_paths[idx])
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            img = self.transform(image=img)['image']
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        clinical = torch.tensor(self.clinical[idx], dtype=torch.float32)
        genomic = torch.tensor(self.genomic[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        mask = torch.ones(3)
        if self.training and self.modality_dropout > 0:
            drop = torch.rand(3) > self.modality_dropout
            if drop.sum() == 0:
                drop[torch.randint(0, 3, (1,))] = True
            mask = drop.float()

        return {'image': img, 'clinical': clinical, 'genomic': genomic, 'label': label, 'modality_mask': mask}

# Transforms
train_transform = A.Compose([
    A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=20, p=0.5),
    A.OneOf([A.GaussNoise(var_limit=(10, 50)), A.GaussianBlur(), A.MotionBlur()], p=0.3),
    A.OneOf([A.RandomBrightnessContrast(), A.HueSaturationValue()], p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Stratified split
indices = np.arange(len(all_labels))
train_val_idx, test_idx = train_test_split(indices, test_size=0.15, stratify=all_labels, random_state=42)
train_idx, val_idx = train_test_split(train_val_idx, test_size=0.18, stratify=all_labels[train_val_idx], random_state=42)

# REVISION: keep test source labels aligned to test_idx order for per-dataset eval
test_source = all_source[test_idx]
np.save("results/test_idx.npy", test_idx)  # reproducibility: exact split indices

print(f" Data Split:")
print(f"   Train: {len(train_idx)} samples")
print(f"   Val:   {len(val_idx)} samples")
print(f"   Test:  {len(test_idx)} samples")

# Create datasets
train_dataset = MultiModalDataset(
    [all_paths[i] for i in train_idx], clinical_X[train_idx], genomic_X[train_idx],
    all_labels[train_idx], train_transform, config.MODALITY_DROPOUT_PROB, True
)
val_dataset = MultiModalDataset(
    [all_paths[i] for i in val_idx], clinical_X[val_idx], genomic_X[val_idx],
    all_labels[val_idx], val_transform, 0.0, False
)
test_dataset = MultiModalDataset(
    [all_paths[i] for i in test_idx], clinical_X[test_idx], genomic_X[test_idx],
    all_labels[test_idx], val_transform, 0.0, False
)

# Weighted sampler for class imbalance
train_labels = all_labels[train_idx]
class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
sample_weights = class_weights[train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler,
                          num_workers=config.NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                         num_workers=config.NUM_WORKERS, pin_memory=True)

print("\n DataLoaders ready!")
