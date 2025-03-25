from .model import ClassificationModel
from .ovd_model import OVDClassificationModel
from .hierarchical_model import HierarchicalClassificationModel
from .convnext import *
from .efficientnet import *
from .image_encoder import ImageEncoder
from .resnet import *
from .resnet_cifar import *
from .bit_resnet import *
from .simple_cnn import SimpleCNN

# All available models and training architectures
__all__ = [
    "ClassificationModel",
    "OVDClassificationModel",
    "HierarchicalClassificationModel",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "ResNet34_CIFAR",
    "ResNet50_CIFAR",
    "ResNet101_CIFAR",
    "ResNet152_CIFAR",
    "ConvNextBase",
    "ConvNextTiny",
    "ConvNextSmall",
    "ConvNextLarge",
    "EfficientNetV2S",
    "EfficientNetV2M",
    "EfficientNetV2L",
    "EfficientNetB0",
    "EfficientNetB7",
    "BitResNet101x1_CIFAR100",
    "BitResNet101x3_CIFAR100",
    "BitResNet50x1_CIFAR",
    "BitResNet50x2_CIFAR",
    "BitResNet50x3_CIFAR",
    "BitResNet101x1_CIFAR",
    "BitResNet101x2_CIFAR",
    "BitResNet101x3_CIFAR",
    "BitResNet152x1_CIFAR",
    "BitResNet152x2_CIFAR",
    "BitResNet152x3_CIFAR",
    "BitResNet50x1_ImageNet",
    "BitResNet50x3_ImageNet",
    "BitResNet101x1_ImageNet",
    "BitResNet101x3_ImageNet",
    "SimpleCNN",
    "ImageEncoder",
]
