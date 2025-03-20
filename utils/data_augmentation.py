import albumentations as A
import argparse
import cv2
import numpy as np
import os
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, ToPILImage
from tqdm import tqdm


auto_augment = AutoAugment(AutoAugmentPolicy.CIFAR10)
def auto_augment_transform(image, **kwargs):
    pil_image = ToPILImage()(image)  
    aug_image = auto_augment(pil_image)
    return np.array(aug_image)
IMAGE_SIZE = 32
TRANSFORMS = A.Compose([
    A.RandomCrop(width=int(IMAGE_SIZE * 0.8), height=int(IMAGE_SIZE * 0.8), p=0.2),
    A.HorizontalFlip(p=0.5),
    A.Lambda(image=auto_augment_transform),
    A.RandomBrightnessContrast(p=0.2),
    A.RGBShift(p=0.2),
    A.Rotate(limit=15, p=0.2),
    A.AdvancedBlur(blur_limit=5, p=0.3),
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.ImageCompression(p=0.1),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
])


def run_aug(image, transforms):
    results = transforms(image=image)
    aug_image = results['image']

    return aug_image


def main(args):
    for category in tqdm(os.listdir(args.image_folder)):
        if category.startswith('.'):
            continue
        category_folder = os.path.join(args.image_folder, category)
        aug_category_folder = os.path.join(args.aug_folder, category)
        if not os.path.exists(aug_category_folder):
            os.makedirs(aug_category_folder)

        for image_name in os.listdir(category_folder):
            if not (image_name.endswith('.jpg') or image_name.endswith('.png')):
                continue
            image_path = os.path.join(category_folder, image_name)
            image = cv2.imread(image_path)

            for i in range(args.aug_size):
                aug_image = run_aug(image, TRANSFORMS)
                image_base_name = image_name.split(".")[0]
                aug_image_path = os.path.join(aug_category_folder, f"{image_base_name}_{i}.png")
                cv2.imwrite(aug_image_path, aug_image)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='./data/cifar100_data/train')
    parser.add_argument('--aug_folder', type=str, default='./data/aug_cifar100_data')
    parser.add_argument('--aug_size', type=int, default=1)
    args = parser.parse_args()

    main(args)
