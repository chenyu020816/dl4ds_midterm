import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from .hierarchical_data_converter import coarse_to_fine
from .utils import class_names, coarse_to_fine, fine_to_coarse




transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])


def build_cifar100_hierarchical_dataloader(config, mode='train', transform=transform):
    coarse_folder = os.path.join(config.DATA_PATH, "coarse_data")
    # fine_folder = os.path.join(config.DATA_PATH, "fine_data")
    assert os.path.exists(coarse_folder), f"{coarse_folder} not exists"
    # assert os.path.exists(fine_folder), f"{fine_folder} not exists"

    coarse_dataset = datasets.ImageFolder(root=coarse_folder, transform=transform)

    # coarse_list = list(coarse_to_fine.keys())

    
    if mode == 'train':
        train_size = int(len(dataset) * config.TRAIN_SIZE)