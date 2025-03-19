from torch import nn

LabelSmooth_CE = nn.CrossEntropyLoss(label_smoothing=0.1)