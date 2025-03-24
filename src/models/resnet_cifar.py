import torch
import torch.nn as nn

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
    return ResNet(Block, [3, 4, 6, 3], num_classes, 3, act=nn.ReLU())


def ResNet50_CIFAR(num_classes, pretrained=False, **kwargs):
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes, 3, act=nn.ReLU())


def ResNet101_CIFAR(num_classes, pretrained=False, **kwargs):
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes, 3, act=nn.ReLU())


def ResNet152_CIFAR(num_classes, pretrained=False, **kwargs):
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes, 3, act=nn.ReLU())


if __name__ == '__main__':
    model = ResNet50_CIFAR(100, 20)
    print(model)
    data = torch.randn(2, 3, 32, 32)
    y = model(data)
    print(y.shape)