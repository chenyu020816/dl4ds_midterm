import torch
import torch.nn as nn

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample is not None:
            identity = self.downsample(x)
        x += identity
        return self.relu(x)


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        return self.relu(x)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, num_channels=3):
        super().__init__()
        self.in_channels = 64

        self.layers = nn.Sequential(
            # -> 64, 112, 112
            nn.Conv2d(num_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            # -> 64, 56, 56
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *self._make_layer(block, num_blocks[0], in_channels=64),
            *self._make_layer(block, num_blocks[1], in_channels=128, stride=2),
            *self._make_layer(block, num_blocks[2], in_channels=256, stride=2),
            *self._make_layer(block, num_blocks[3], in_channels=512, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * block.expansion, num_classes)
        )

    def _make_layer(self, block, num_blocks, in_channels, stride=1):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != in_channels*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, in_channels*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(in_channels*block.expansion),
            )
        layers.append(block(self.in_channels, in_channels, downsample, stride))
        self.in_channels = in_channels*block.expansion

        for i in range(1, num_blocks):
            layers.append(block(self.in_channels, in_channels))
        return layers

    def forward(self, x):
        return self.layers(x)


def ResNet18(num_classes=100, pretrained=False):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
    if not pretrained:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        for name, param in model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    return model


def ResNet34(num_classes=100, pretrained=False):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=pretrained)
    if not pretrained:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        for name, param in model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    return model


def ResNet50(num_classes=100, pretrained=False):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
    if not pretrained:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        for name, param in model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    return model


def ResNet101(num_classes=100, pretrained=False):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=pretrained)
    if not pretrained:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        for name, param in model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    return model


def ResNet152(num_classes=100, pretrained=False):
    model = torch.hub.load('pytorch/vision', 'resnet152',  weights="IMAGENET1K_V2")
    if not pretrained:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        for name, param in model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    return model


def main():
    model = ResNet152(100, pretrained=True)
    print(model)
    data = torch.randn(2, 3, 32, 32)
    y = model(data)
    print(y.shape)

if __name__ == '__main__':
    main()