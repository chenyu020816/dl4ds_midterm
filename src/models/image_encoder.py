import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(self, base_model, embedding_dim=512):
        super(ImageEncoder, self).__init__()
        self.backbone = base_model

    def forward(self, x):
        x = self.backbone(x)
        return F.normalize(x, dim=-1)