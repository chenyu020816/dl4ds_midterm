import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=100, pretrained=False):
        super(SimpleCNN, self).__init__()
        self.in_channels = 3
        self.layers1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.SiLU(),
            nn.Dropout(0.2),
        )
        self.layers2 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.SiLU(),
        )
        self.ca = ChannelAttention(2048)
        self.spa = SpatialAttention()
        self.layers3 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        )

        self.layers4 = nn.Flatten()
        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.ca(x) * x
        x = self.spa(x) * x
        x = self.layers3(x)
        x = self.layers4(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    import torch
    x = torch.randn(10, 3, 32, 32)
    model = SimpleCNN()
    output = model(x)

