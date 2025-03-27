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
    "ResNet18",                             # Original ResNet18         (w/wo pretrained on ImageNet1K)
    "ResNet34",                             # Original ResNet34         (w/wo pretrained on ImageNet1K)
    "ResNet50",                             # Original ResNet50         (w/wo pretrained on ImageNet1K)
    "ResNet101",                            # Original ResNet101        (w/wo pretrained on ImageNet1K)
    "ResNet152",                            # Original ResNet152        (w/wo pretrained on ImageNet1K)
    "ResNet34_CIFAR",                       # Modified ResNet34         (w/wo pretrained on ImageNet1K)
    "ResNet50_CIFAR",                       # Modified ResNet50         (w/wo pretrained on ImageNet1K)
    "ResNet101_CIFAR",                      # Modified ResNet101        (w/wo pretrained on ImageNet1K)
    "ResNet152_CIFAR",                      # Modified ResNet152        (w/wo pretrained on ImageNet1K)
    "ConvNextBase",                         # Original ConvNextBase     (w/wo pretrained on ImageNet1K)
    "ConvNextTiny",                         # Original ConvNextTiny     (w/wo pretrained on ImageNet1K)
    "ConvNextSmall",                        # Original ConvNextSmall    (w/wo pretrained on ImageNet1K)
    "ConvNextLarge",                        # Original ConvNextLarge    (w/wo pretrained on ImageNet1K)
    "EfficientNetV2S",                      # Original EfficientNetV2S  (w/wo pretrained on ImageNet1K)
    "EfficientNetV2M",                      # Original EfficientNetV2M  (w/wo pretrained on ImageNet1K)
    "EfficientNetV2L",                      # Original EfficientNetV2L  (w/wo pretrained on ImageNet1K)
    "EfficientNetB0",                       # Original EfficientNetB0   (w/wo pretrained on ImageNet1K)
    "EfficientNetB1",                       # Original EfficientNetB1   (w/wo pretrained on ImageNet1K)
    "EfficientNetB2",                       # Original EfficientNetB2   (w/wo pretrained on ImageNet1K)
    "EfficientNetB3",                       # Original EfficientNetB3   (w/wo pretrained on ImageNet1K)
    "EfficientNetB4",                       # Original EfficientNetB4   (w/wo pretrained on ImageNet1K)
    "EfficientNetB5",                       # Original EfficientNetB5   (w/wo pretrained on ImageNet1K)
    "EfficientNetB6",                       # Original EfficientNetB6   (w/wo pretrained on ImageNet1K)
    "EfficientNetB7",                       # Original EfficientNetB7   (w/wo pretrained on ImageNet1K)
    "BitResNet101x1_CIFAR100",              # Original BitResNet101x1   (w/wo pretrained on Cifar100)
    "BitResNet101x3_CIFAR100",              # Original BitResNet101x3   (w/wo pretrained on Cifar100)
    "BitResNet50x1_CIFAR",                  # Modified BitResNet50x1    (wo pretrained)
    "BitResNet50x2_CIFAR",                  # Modified BitResNet50x2    (wo pretrained)
    "BitResNet50x3_CIFAR",                  # Modified BitResNet50x3    (wo pretrained)
    "BitResNet101x1_CIFAR",                 # Modified BitResNet101x1   (wo pretrained)
    "BitResNet101x2_CIFAR",                 # Modified BitResNet101x2   (wo pretrained)
    "BitResNet101x3_CIFAR",                 # Modified BitResNet101x3   (wo pretrained)
    "BitResNet152x1_CIFAR",                 # Modified BitResNet152x1   (wo pretrained)
    "BitResNet152x2_CIFAR",                 # Modified BitResNet152x2   (wo pretrained)
    "BitResNet152x4_CIFAR",                 # Modified BitResNet152x3   (wo pretrained)
    "BitResNet50x1_ImageNet",               # Original BitResNet101x3   (w/wo pretrained on ImageNet1K)
    "BitResNet50x3_ImageNet",               # Original BitResNet101x3   (w/wo pretrained on ImageNet1K)
    "BitResNet101x1_ImageNet",              # Original BitResNet101x3   (w/wo pretrained on ImageNet1K)
    "BitResNet101x3_ImageNet",              # Original BitResNet101x3   (w/wo pretrained on ImageNet1K)
    "SimpleCNN",                            # Simple CNN                (wo pretrained)
]
