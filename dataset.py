import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Callable, List, Tuple
from PIL import Image
from utils import import_data_and_show_summary

class SegmentationTransform:
    def __init__(self, input_shape=(224, 160)):
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(input_shape),
            transforms.RandomRotation(5),  # Smaller rotation for medical images
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),  # Subtle intensity augmentation
            transforms.RandomAffine(
                degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0], std=[0.1]),  # Normalization for CT scans (adjust based on dataset)
        ])

        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(input_shape, interpolation=Image.NEAREST),  # Nearest interpolation to keep mask values intact
            transforms.RandomRotation(5),  # Keep rotation consistent with image
            transforms.RandomAffine(
                degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5
            ),
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