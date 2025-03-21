import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])


def build_cifar100_dataloader(config, mode='train', transform=transform):
    data_folder = os.path.join(config.DATA_PATH)
    assert os.path.isdir(data_folder), f"{data_folder} is not exists"
    dataset = datasets.ImageFolder(root=data_folder, transform=transform)

    if mode == 'train':
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
            transform=transform
        )
        test_loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKER
        )
        return test_loader
