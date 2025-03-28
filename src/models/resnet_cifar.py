import torch
import torch.nn as nn
from torchvision import models


def get_pretrained_resnet_cifar(model_name, num_classes):
    model_name = model_name.lower()

    # using models.resnet(pretrained=True) will fail on some version of pytorch
    if model_name == 'resnet18':
        pretrained_weights = models.ResNet18_Weights.IMAGENET1K_V1 
        model = models.resnet18(weights=pretrained_weights)
    elif model_name == 'resnet34':
        pretrained_weights = models.ResNet34_Weights.IMAGENET1K_V1 
        model = models.resnet34(weights=pretrained_weights)
    elif model_name == 'resnet50':
        pretrained_weights = models.ResNet50_Weights.IMAGENET1K_V1 
        model = models.resnet50(weights=pretrained_weights)
    elif model_name == 'resnet101':
        pretrained_weights = models.ResNet101_Weights.IMAGENET1K_V1
        model = models.resnet101(weights=pretrained_weights)
    elif model_name == 'resnet152':
        pretrained_weights = models.ResNet152_Weights.IMAGENET1K_V1
        model = models.resnet152(weights=pretrained_weights)
    else:
        ValueError(f"Invalid model name: {model_name}")
        return

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # freeze all parameters except last layer
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "layer4" in name:
            param.requires_grad = True
        elif "conv1" in name:
            param.requires_grad = True
   
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model


class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, downsample=None, stride=1, act=nn.ReLU()):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.act = act

    def forward(self, x):
        identity = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample is not None:
            identity = self.downsample(x)
        x += identity
        return self.act(x)


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, downsample=None, stride=1, act=nn.ReLU()):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.downsample = downsample
        self.act = act

    def forward(self, x):
        identity = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        return self.act(x)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_channels=3, act=nn.ReLU()):
        super().__init__()
        self.in_channels = 64
        self.act = act
        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *self._make_layer(block, num_blocks[0], in_channels=64, act=self.act),
            *self._make_layer(block, num_blocks[1], in_channels=128, stride=2, act=self.act),
            *self._make_layer(block, num_blocks[2], in_channels=256, stride=2, act=self.act),
            *self._make_layer(block, num_blocks[3], in_channels=512, stride=1, act=self.act),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * block.expansion, num_classes)
        )

    def _make_layer(self, block, num_blocks, in_channels, stride=1, act=nn.ReLU()):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != in_channels*block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.in_channels, in_channels*block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(in_channels*block.expansion),
            )
        layers.append(block(self.in_channels, in_channels, downsample, stride, act=act))
        self.in_channels = in_channels*block.expansion

        for i in range(1, num_blocks):
            layers.append(block(self.in_channels, in_channels))
        return layers

    def forward(self, x):
        return self.layers(x)


def ResNet34_CIFAR(num_classes, pretrained=False, **kwargs):
    if not pretrained:
        return ResNet(Block, [3, 4, 6, 3], num_classes, 3, act=nn.ReLU())
    else:
        return get_pretrained_resnet_cifar("resnet34", num_classes)


def ResNet50_CIFAR(num_classes, pretrained=False, **kwargs):
    if not pretrained:
        return ResNet(BottleNeck, [3, 4, 6, 3], num_classes, 3, act=nn.ReLU())
    else:
        return get_pretrained_resnet_cifar("resnet50", num_classes)


def ResNet101_CIFAR(num_classes, pretrained=False, **kwargs):
    if not pretrained:
        return ResNet(BottleNeck, [3, 4, 23, 3], num_classes, 3, act=nn.ReLU())
    else:
        return get_pretrained_resnet_cifar("resnet101", num_classes)


def ResNet152_CIFAR(num_classes, pretrained=False, **kwargs):
    if not pretrained:
        return ResNet(BottleNeck, [3, 8, 36, 3], num_classes, 3, act=nn.ReLU())
    else:
        return get_pretrained_resnet_cifar("resnet152", num_classes)


if __name__ == '__main__':
    model = ResNet34_CIFAR(100, True)
    print(model)
    data = torch.randn(2, 3, 32, 32)
    y = model(data)
    print(y.shape)