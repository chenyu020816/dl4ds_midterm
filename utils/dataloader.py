import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])


def build_cifar100_dataloader(config, data_path, mode='train', transform=base_transform):
    if mode == 'train':
        data_folder = os.path.join(data_path)
        assert os.path.isdir(data_folder), f"{data_folder} is not exists"
        dataset = datasets.ImageFolder(root=data_folder, transform=transform)
        train_size = int(len(dataset) * config.TRAIN_SIZE)
        val_size = len(dataset) - train_size
        train_data, val_data = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(
            train_data,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS)
        val_loader = DataLoader(
            val_data,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS)
        return train_loader, val_loader
    else:
        dataset = datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=base_transform
        )
        test_loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS
        )
        return test_loader


def build_transform(transform_config):
    base_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])

    if transform_config.AUTOAUG:
        base_transform.transforms.append(
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)
        )
    base_transform.transforms.append(transforms.ToTensor())
    if transform_config.ERASE_PROB > 0:
        base_transform.transforms.append(
            transforms.RandomErasing(
                p=transform_config.ERASE_PROB,
                scale=(0.0625, 0.1),
                ratio=(0.99, 1.0)
            )
        )
    base_transform.transforms.append(
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    )

    return base_transform