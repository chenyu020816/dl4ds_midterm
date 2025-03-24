import torch
import torch.nn as nn
from torchvision import models

def get_efficientnet(model_name, num_classes, pretrained):
    # model_name = model_name.lower()

    if model_name == 'efficientnet_v2s':
        pretrained_weights = models.EfficientNet_V2_S_Weights if pretrained else None
        model = models.efficientnet_v2_s(weights=pretrained_weights)
    elif model_name == 'efficientnet_v2m':
        pretrained_weights = models.EfficientNet_V2_M_Weights if pretrained else None
        model = models.efficientnet_v2_m(weights=pretrained_weights)
    elif model_name == 'efficientnet_v2l':
        pretrained_weights = models.EfficientNet_V2_L_Weights if pretrained else None
        model = models.efficientnet_v2_l(weights=pretrained_weights)
    elif model_name == 'efficientnet_b0':
        pretrained_weights = models.EfficientNet_B0_Weights if pretrained else None
        model = models.efficientnet_b0(weights=pretrained_weights)
    elif model_name == 'efficientnet_b1':
        pretrained_weights = models.EfficientNet_B1_Weights if pretrained else None
        model = models.efficientnet_b1(weights=pretrained_weights)
    elif model_name == 'efficientnet_b2':
        pretrained_weights = models.EfficientNet_B2_Weights if pretrained else None
        model = models.efficientnet_b2(weights=pretrained_weights)
    elif model_name == 'efficientnet_b3':
        pretrained_weights = models.EfficientNet_B3_Weights if pretrained else None
        model = models.efficientnet_b3(weights=pretrained_weights)
    elif model_name == 'efficientnet_b4':
        pretrained_weights = models.EfficientNet_B4_Weights if pretrained else None
        model = models.efficientnet_b4(weights=pretrained_weights)
    elif model_name == 'efficientnet_b5':
        pretrained_weights = models.EfficientNet_B5_Weights if pretrained else None
        model = models.efficientnet_b5(weights=pretrained_weights)
    elif model_name == 'efficientnet_b6':
        pretrained_weights = models.EfficientNet_B6_Weights if pretrained else None
        model = models.efficientnet_b6(weights=pretrained_weights)
    elif model_name == 'efficientnet_b7':
        pretrained_weights = models.EfficientNet_B7_Weights if pretrained else None
        model = models.efficientnet_b7(weights=pretrained_weights)
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
            if "features.6" in name or "features.7" in name:
                param.requires_grad = True

            # Modify the classifier
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    return model


def EfficientNetV2S(num_classes, pretrained=False):
    return get_efficientnet('efficientnet_v2s', num_classes, pretrained)


def EfficientNetV2M(num_classes, pretrained=False):
    return get_efficientnet('efficientnet_v2m', num_classes, pretrained)


def EfficientNetV2L(num_classes, pretrained=False):
        return get_efficientnet('efficientnet_v2l', num_classes, pretrained)


def EfficientNetB0(num_classes, pretrained=False):
    return get_efficientnet('efficientnet_b0', num_classes, pretrained)


def EfficientNetB1(num_classes, pretrained=False):
    return get_efficientnet('efficientnet_b1', num_classes, pretrained)


def EfficientNetB2(num_classes, pretrained=False):
    return get_efficientnet('efficientnet_b2', num_classes, pretrained)


def EfficientNetB3(num_classes, pretrained=False):
    return get_efficientnet('efficientnet_b3', num_classes, pretrained)


def EfficientNetB4(num_classes, pretrained=False):
    return get_efficientnet('efficientnet_b4', num_classes, pretrained)


def EfficientNetB5(num_classes, pretrained=False):
    return get_efficientnet('efficientnet_b5', num_classes, pretrained)


def EfficientNetB6(num_classes, pretrained=False):
    return get_efficientnet('efficientnet_b6', num_classes, pretrained)


def EfficientNetB7(num_classes, pretrained=False):
    return get_efficientnet('efficientnet_b7', num_classes, pretrained)


if __name__ == '__main__':
    import torch
    # test model input & output
    model = EfficientNetV2M(100, pretrained=True)
    print(model)
    data = torch.randn(2, 3, 32, 32)
    y = model(data)
    print(y.shape)