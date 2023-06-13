import torchvision.transforms as transform
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.datasets.folder import default_loader

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import numpy as np
import pandas as pd
import os
import torch


class SequenceDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transforms=None, target_transforms=None):
        super().__init__()
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transforms
        self.target_transform = target_transforms

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, str(img_path))
        image = default_loader(img_path)
        target = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.img_labels.index)


def custom_dataloader(csv_path, image_directiory, batch_size, t=None, num_workers=6):
    if t is None:
        t = transform.Compose([transform.ToTensor(), transform.Resize((224, 224), interpolation=BICUBIC), transform.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    d = SequenceDataset(csv_path, image_directiory, transforms=t)

    dataset_size = len(d)
    print("Dataset size:", dataset_size)

    # Subset random sampler dataloader: not weighted
    index_list = list(range(dataset_size))

    # sampler for both dataloaders
    train_sampler = SubsetRandomSampler(index_list)

    train_loader = DataLoader(d, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)

    return {'train': train_loader}


if __name__ == "__main__":
    dataloaders_dict, class_weight = custom_dataloader()
