"""Architecture: ensemble image encoder (ResNet50 + DenseNet121 + EfficientNet-B0),
clinical/genomic encoders, Monte Carlo Dropout wrappers, and the uncertainty-aware
adaptive fusion operator (convex combination over the class-probability simplex,
per-sample weights from MC-Dropout predictive entropy).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ImageEncoder(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        self.resnet = timm.create_model('resnet50', pretrained=True, num_classes=0)
        self.densenet = timm.create_model('densenet121', pretrained=True, num_classes=0)
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)

        total_dim = self.resnet.num_features + self.densenet.num_features + self.efficientnet.num_features

        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, return_features=False):
        f1 = self.resnet(x)
        f2 = self.densenet(x)
        f3 = self.efficientnet(x)
        features = self.fusion(torch.cat([f1, f2, f3], dim=1))
        logits = self.classifier(features)
        return (logits, features) if return_features else logits

class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim, num_classes=2, dropout=0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x, return_features=False):
        features = self.encoder(x)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits

class GenomicEncoder(nn.Module):
    def __init__(self, input_dim, num_classes=2, dropout=0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, return_features=False):
        features = self.encoder(x)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits

class MCDropoutWrapper(nn.Module):
    def __init__(self, model, n_samples=20):
        super().__init__()
        self.model = model
        self.n_samples = n_samples

    def enable_dropout(self):
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def forward(self, x, return_uncertainty=False, return_features=False):
        if not return_uncertainty:
            return self.model(x, return_features=return_features)

        self.enable_dropout()
        preds = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                out = self.model(x)
                preds.append(F.softmax(out, dim=-1))

        preds = torch.stack(preds, dim=0)
        mean_pred = preds.mean(dim=0)
        uncertainty = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=-1) / np.log(mean_pred.shape[-1])
        return mean_pred, uncertainty

class UncertaintyAwareFusion(nn.Module):
    def __init__(self, n_modalities=3, num_classes=2):
        super().__init__()
        self.base_weights = nn.Parameter(torch.ones(n_modalities) / n_modalities)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, predictions, uncertainties, modality_mask):
        certainty = 1.0 - uncertainties.clamp(0, 1)
        weighted = certainty * F.softplus(self.base_weights).unsqueeze(0) * modality_mask
        fusion_weights = F.softmax(weighted / (self.temperature.abs() + 0.1), dim=1)
        fused = (predictions * fusion_weights.unsqueeze(-1)).sum(dim=1)
        fused = fused / (fused.sum(dim=-1, keepdim=True) + 1e-10)
        confidence = (certainty * fusion_weights).sum(dim=1)
        return fused, fusion_weights, confidence

class MultiModalModel(nn.Module):
    def __init__(self, clinical_dim, genomic_dim, num_classes=2, dropout=0.4, mc_samples=20):
        super().__init__()
        self.image_encoder = ImageEncoder(num_classes, dropout)
        self.clinical_encoder = ClinicalEncoder(clinical_dim, num_classes, dropout)
        self.genomic_encoder = GenomicEncoder(genomic_dim, num_classes, dropout)

        self.image_mc = MCDropoutWrapper(self.image_encoder, mc_samples)
        self.clinical_mc = MCDropoutWrapper(self.clinical_encoder, mc_samples)
        self.genomic_mc = MCDropoutWrapper(self.genomic_encoder, mc_samples)

        self.fusion = UncertaintyAwareFusion(3, num_classes)

    def forward(self, image, clinical, genomic, mask, return_uncertainty=False, return_details=False):
        B = image.shape[0]

        if return_uncertainty:
            img_pred, img_unc = self.image_mc(image, return_uncertainty=True)
            clin_pred, clin_unc = self.clinical_mc(clinical, return_uncertainty=True)
            gen_pred, gen_unc = self.genomic_mc(genomic, return_uncertainty=True)
            preds = torch.stack([img_pred, clin_pred, gen_pred], dim=1)
            uncs = torch.stack([img_unc, clin_unc, gen_unc], dim=1)
        else:
            img_pred = F.softmax(self.image_encoder(image), dim=-1)
            clin_pred = F.softmax(self.clinical_encoder(clinical), dim=-1)
            gen_pred = F.softmax(self.genomic_encoder(genomic), dim=-1)
            preds = torch.stack([img_pred, clin_pred, gen_pred], dim=1)
            uncs = torch.ones(B, 3, device=image.device) * 0.1

        preds = preds * mask.unsqueeze(-1)
        fused, weights, conf = self.fusion(preds, uncs, mask)

        if return_details:
            return {
                'prediction': fused, 'fusion_weights': weights, 'confidence': conf,
                'modality_preds': preds, 'modality_uncs': uncs
            }
        return fused, conf
