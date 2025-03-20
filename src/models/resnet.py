import torch
import torch.nn as nn


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


if __name__ == '__main__':
    # test model input & output
    model = ResNet152(100, pretrained=True)
    print(model)
    data = torch.randn(2, 3, 32, 32)
    y = model(data)
    print(y.shape)