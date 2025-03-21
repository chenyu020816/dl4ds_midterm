import torch
import torch.nn as nn
from torchvision import models


def get_convnext(model_name, num_classes, pretrained):
    model_name = model_name.lower()

    if model_name == 'convnext_base':
        pretrained_weights = models.ConvNeXt_Base_Weights if pretrained else None
        model = models.convnext_base(weights=pretrained_weights)
    elif model_name == 'convnext_tiny':
        pretrained_weights = models.ConvNeXt_Tiny_Weights if pretrained else None
        model = models.convnext_tiny(weights=pretrained_weights)
    elif model_name == 'convnext_small':
        pretrained_weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_small(weights=pretrained_weights)
    elif model_name == 'convnext_large':
        pretrained_weights = not models.ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_large(weights=pretrained_weights)
    else:
        ValueError(f"Invalid model name: {model_name}")
        return

    if not pretrained:
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    else:
        # freeze all parameters except last layer
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            if "features.7" in name:
                param.requires_grad = True

        model.classifier[2] = nn.Sequential(
            nn.Linear(model.classifier[2].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    return model


def ConvNextBase(num_classes, pretrained=False):
    return get_convnext("convnext_base", num_classes, pretrained)


def ConvNextTiny(num_classes, pretrained=False):
    return get_convnext("convnext_tiny", num_classes, pretrained)


def ConvNextSmall(num_classes, pretrained=False):
    return get_convnext("convnext_small", num_classes, pretrained)


def ConvNextMedium(num_classes, pretrained=False):
    return get_convnext("convnext_medium", num_classes, pretrained)


def ConvNextLarge(num_classes, pretrained=False):
    return get_convnext("convnext_large", num_classes, pretrained)


if __name__ == '__main__':
    # test model input & output
    model = ConvNextSmall(100, pretrained=True)
    print(model)
    data = torch.randn(2, 3, 32, 32)
    y = model(data)
    print(y.shape)