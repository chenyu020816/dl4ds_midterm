import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ArcFaceLoss(nn.Module):
    def __init__(self, margin=0.5, scale=64.0):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, logits, labels):
        cos_theta = F.linear(F.normalize(logits), F.normalize(logits))
        theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
        target_logits = torch.cos(theta + self.margin)
        return F.cross_entropy(self.scale * target_logits, labels)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features, labels):
        logits = (image_features @ text_features.T) / self.temperature
        labels = labels.to(logits.device)
        return F.cross_entropy(logits, labels)


CE = nn.CrossEntropyLoss()
LabelSmooth_CE = nn.CrossEntropyLoss(label_smoothing=0.1)
FC = FocalLoss(alpha=0.25, gamma=2.0)
AF = ArcFaceLoss(margin=0.5, scale=64.0)
CL = ContrastiveLoss()