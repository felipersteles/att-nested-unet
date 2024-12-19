import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Callable, List, Tuple
from PIL import Image
from models.utils import import_data_and_show_summary
class SegmentationTransform:
    def __init__(self, input_shape=(224, 160)):
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_shape),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),  # Smaller rotation for medical images
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Subtle intensity augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0], std=[0.1]),  # Normalization for CT scans (adjust based on dataset)
        ])

        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_shape, interpolation=Image.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),  # Keep rotation consistent with image
            transforms.ToTensor(),
        ])


    def __call__(self, image, mask):
        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        return image, mask


class PancreasDataset(Dataset):
    def __init__(self, data_dir=None, num_patients=None, transform: Callable = None):
        if data_dir is None:
            raise ValueError("Data directory must be provided.")
        
        images, labels = import_data_and_show_summary(data_dir, num_patients)

        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image, label = self.transform(image, label)

        return image, label