import os
from torch.utils.data import random_split
from torchvision import datasets, transforms
import json


def main(output_dir = "./data/cifar100_data_yolo"):
    os.makedirs(output_dir, exist_ok=True)

    def create_class_dirs(class_names):
        for class_name in class_names:
            os.makedirs(os.path.join(output_dir, "train", class_name), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "test", class_name), exist_ok=True)

    cifar100_classes = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
        'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
        'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
        'sunflower',
        'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
        'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]

    create_class_dirs(cifar100_classes)

    def save_cifar_images(dataset, dataset_type="train"):
        labels_dict = {}
        for idx, (image, label) in enumerate(dataset):
            class_name = cifar100_classes[label]
            image_path = os.path.join(output_dir, dataset_type, class_name, f"{idx}.png")
            image.save(image_path)
            labels_dict[image_path] = class_name

        with open(os.path.join(output_dir, f"{dataset_type}_labels.json"), "w") as f:
            json.dump(labels_dict, f, indent=4)

    dataset = datasets.CIFAR100(root="./data", train=True, download=True)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    save_cifar_images(train_data, dataset_type="train")
    save_cifar_images(val_data, dataset_type="test")


if __name__ == "__main__":
    main()