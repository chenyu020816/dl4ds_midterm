import torch
import torch.nn as nn
from torchvision import models

def get_resnet(model_name, num_classes, pretrained):
    model_name = model_name.lower()

    # using models.resnet(pretrained=True) will fail on some version of pytorch
    if model_name == 'resnet18':
        pretrained_weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=pretrained_weights)
    elif model_name == 'resnet34':
        pretrained_weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=pretrained_weights)
    elif model_name == 'resnet50':
        pretrained_weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=pretrained_weights)
    elif model_name == 'resnet101':
        pretrained_weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet101(weights=pretrained_weights)
    elif model_name == 'resnet152':
        pretrained_weights = models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet152(weights=pretrained_weights)
    else:
        ValueError(f"Invalid model name: {model_name}")
        return

    if not pretrained:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        # freeze all parameters except last layer
        for param in model.parameters():
            param.requires_grad = False

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

def ResNet18(num_classes=100, pretrained=False):
    return get_resnet('resnet18', num_classes, pretrained)


def ResNet34(num_classes=100, pretrained=False):
    return get_resnet('resnet34', num_classes, pretrained)


def ResNet50(num_classes=100, pretrained=False):
    return get_resnet('resnet50', num_classes, pretrained)


def ResNet101(num_classes=100, pretrained=False):
    return get_resnet('resnet101', num_classes, pretrained)


def ResNet152(num_classes=100, pretrained=False):
    return get_resnet('resnet152', num_classes, pretrained)


if __name__ == '__main__':
    # test model input & output
    model = ResNet34(100, pretrained=False)
    print(model)
    data = torch.randn(2, 3, 32, 32)
    y = model(data)
    print(y.shape)

    model = ResNet34(100, pretrained=True)
    print(model)
    data = torch.randn(2, 3, 32, 32)
    y = model(data)
    print(y.shape)

