import torch
import numpy as np
import random
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Callable, List, Tuple
from PIL import Image
from models.utils import import_data_and_show_summary
from torchvision.transforms import InterpolationMode

class SegmentationTransform:
    def __init__(self, input_shape=(224, 160)):
        self.input_shape = input_shape

        # Transformations for the image
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
            transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.5),
            transforms.RandomResizedCrop(input_shape, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomRotation(degrees=10),  # Slightly increased rotation
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # Larger translation
                scale=(0.9, 1.1),
                shear=(5, 5),  # Allow more shear
            ),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),  # Blur for additional augmentation
            transforms.Resize(input_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[-1000.0], std=[2000.0]),  # CT-specific normalization (adjust as needed)
        ])

        # Transformations for the mask
        self.mask_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
            transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.5),
            transforms.RandomResizedCrop(input_shape, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=InterpolationMode.NEAREST),
            transforms.RandomRotation(degrees=10, interpolation=Image.NEAREST),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=(5, 5),
                interpolation=InterpolationMode.NEAREST,
            ),
            transforms.Resize(input_shape, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __call__(self, image, mask):
        # Apply random seed for consistent transformation between image and mask
        seed = random.randint(0, 2**32 - 1)

        # Apply the same seed for deterministic behavior
        random.seed(seed)
        image = self.image_transform(image)

        random.seed(seed)
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