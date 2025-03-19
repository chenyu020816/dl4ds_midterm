import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=100):
        super(SimpleCNN, self).__init__()
        self.in_channels = in_channels
        self.layers1 =  nn.Sequential(
            nn.Conv2d(self.in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.layers2 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.layers3 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc1 = nn.Linear(4608, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x