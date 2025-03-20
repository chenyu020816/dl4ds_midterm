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


CE = nn.CrossEntropyLoss()
LabelSmooth_CE = nn.CrossEntropyLoss(label_smoothing=0.1)
FC = FocalLoss(alpha=0.25, gamma=2.0)
AF = ArcFaceLoss(margin=0.5, scale=64.0)