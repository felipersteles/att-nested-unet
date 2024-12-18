import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Callable, List, Tuple
from PIL import Image
from models.utils import import_data_and_show_summary

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class SegmentationTransform:
    def __init__(self, input_shape=(224, 160)):
        self.transform = A.Compose(
            [
                # Spatial augmentations
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.95, 1.05),
                    translate_percent=(0.05, 0.05),
                    rotate=(-5, 5),
                    shear=(-5, 5),
                    p=0.8
                ),
                A.Resize(height=input_shape[0], width=input_shape[1]),
                
                # Intensity augmentations
                A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),  # Simulate subtle motion blur
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),  # Add Gaussian noise
                
                # Convert to PyTorch tensor
                ToTensorV2(),
            ]
        )

        # Mask-specific transformations (must align spatially with image)
        self.mask_transform = A.Compose(
            [
                A.Resize(height=input_shape[0], width=input_shape[1]),
                ToTensorV2(),
            ]
        )

    def __call__(self, image, mask):
        # Ensure consistent random augmentations for both image and mask
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]
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