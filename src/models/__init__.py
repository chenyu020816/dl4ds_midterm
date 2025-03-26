from .model import ClassificationModel
from .ovd_model import OVDClassificationModel
from .hierarchical_model import HierarchicalClassificationModel
from .convnext import *
from .efficientnet import *
from .image_encoder import ImageEncoder
from .resnet import *
from .resnet_cifar import *
from .densenet import *
from .bit_resnet import *
from .simple_cnn import SimpleCNN

# All available models and training architectures
__all__ = [
    "ClassificationModel",                  # not available model
    "OVDClassificationModel",               # not available model
    "HierarchicalClassificationModel",      # not available model
    "ImageEncoder",                         # not available model
    "DenseNet121_CIFAR",                    # Modified DenseNet121      (wo pretrained)
    "DenseNet161_CIFAR",                    # Modified DenseNet161      (wo pretrained)
    "DenseNet169_CIFAR",                    # Modified DenseNet169      (wo pretrained)
    "DenseNet201_CIFAR",                    # Modified DenseNet201      (wo pretrained)
    "ResNet18",                             # Original ResNet18         (w/wo pretrained)
    "ResNet34",                             # Original ResNet34         (w/wo pretrained)
    "ResNet50",                             # Original ResNet50         (w/wo pretrained)
    "ResNet101",                            # Original ResNet101        (w/wo pretrained)
    "ResNet152",                            # Original ResNet152        (w/wo pretrained)
    "ResNet34_CIFAR",                       # Modified ResNet34         (wo pretrained)
    "ResNet50_CIFAR",                       # Modified ResNet34         (wo pretrained)
    "ResNet101_CIFAR",                      # Modified ResNet34         (wo pretrained)
    "ResNet152_CIFAR",                      # Modified ResNet34         (wo pretrained)
    "ConvNextBase",
    "ConvNextTiny",
    "ConvNextSmall",
    "ConvNextLarge",
    "EfficientNetV2S",
    "EfficientNetV2M",
    "EfficientNetV2L",
    "EfficientNetB0",
    "EfficientNetB1",
    "EfficientNetB2",
    "EfficientNetB3",
    "EfficientNetB4",
    "EfficientNetB5",
    "EfficientNetB6",
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
]
