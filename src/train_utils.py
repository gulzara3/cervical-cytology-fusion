"""Focal loss, early stopping, metric tracking, and train/eval loops."""
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

from .config import config


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce
        return focal_loss.mean()

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class MetricTracker:
    def __init__(self):
        self.history = defaultdict(list)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.history[k].append(v)

    def get(self, key):
        return self.history[key]

def train_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Training", leave=False):
        img = batch['image'].to(device)
        clin = batch['clinical'].to(device)
        gen = batch['genomic'].to(device)
        labels = batch['label'].to(device)
        mask = batch['modality_mask'].to(device)

        optimizer.zero_grad()

        with autocast(enabled=config.USE_AMP):
            pred, _ = model(img, clin, gen, mask)
            loss = criterion(pred, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        all_preds.extend(pred.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), accuracy_score(all_labels, all_preds)

@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds=False):
    model.eval()
    total_loss = 0
    all_preds, all_probs, all_labels = [], [], []
    all_confs, all_weights = [], []
    all_mod_preds, all_mod_uncs = [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        img = batch['image'].to(device)
        clin = batch['clinical'].to(device)
        gen = batch['genomic'].to(device)
        labels = batch['label'].to(device)
        mask = batch['modality_mask'].to(device)

        out = model(img, clin, gen, mask, return_uncertainty=True, return_details=True)
        pred = out['prediction']

        loss = criterion(pred, labels)
        total_loss += loss.item()

        all_probs.extend(pred.cpu().numpy())
        all_preds.extend(pred.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confs.extend(out['confidence'].cpu().numpy())
        all_weights.extend(out['fusion_weights'].cpu().numpy())
        all_mod_preds.extend(out['modality_preds'].cpu().numpy())
        all_mod_uncs.extend(out['modality_uncs'].cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    metrics = {
        'loss': total_loss / len(loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs[:, 1]) if len(np.unique(all_labels)) > 1 else 0.5,
        'confidence': np.mean(all_confs)
    }

    if return_preds:
        return metrics, {
            'labels': all_labels, 'predictions': all_preds, 'probabilities': all_probs,
            'confidences': np.array(all_confs), 'fusion_weights': np.array(all_weights),
            'modality_preds': np.array(all_mod_preds), 'modality_uncs': np.array(all_mod_uncs)
        }
    return metrics
