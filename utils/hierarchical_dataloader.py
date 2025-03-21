import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from .hierarchical_data_converter import coarse_to_fine

fine_to_coarse = {fine: coarse for coarse, fine_list in coarse_to_fine.items() for fine in fine_list}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])


class CoarseDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.coarse_labels = [fine_to_coarse[self.classes[idx]] for idx in self.targets]
        self.coarse_classes = list(coarse_to_fine.keys())  # 大類別清單

    def __getitem__(self, index):
        img, fine_label = super().__getitem__(index)
        coarse_label = self.coarse_classes.index(self.coarse_labels[index])
        return img, coarse_label