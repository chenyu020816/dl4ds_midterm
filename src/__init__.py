from .models import *
from .criterion import *
from .early_stopping import EarlyStopping


__all__ = [
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "SimpleCNN",
    "CE",
    "LabelSmooth_CE",
    "EarlyStopping",
]


CE = nn.CrossEntropyLoss()
LabelSmooth_CE = nn.CrossEntropyLoss(label_smoothing=0.1)
FC = FocalLoss(alpha=0.25, gamma=2.0)
AF = ArcFaceLoss(margin=0.5, scale=64.0)