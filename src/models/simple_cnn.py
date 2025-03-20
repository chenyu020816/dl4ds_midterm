import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=100):
        super(SimpleCNN, self).__init__()
        self.in_channels = in_channels
        self.layers1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.SiLU(),
        )
        self.layers2 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        )
        self.layers3 = nn.Flatten()
        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    import torch
    x = torch.randn(10, 3, 32, 32)
    model = SimpleCNN()
    output = model(x)

