import os
import torch
import cv2
import numpy as np
import json
from torchvision import transforms
from PIL import Image
from torch import nn
from torchvision.models import resnet50


class CTTransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), d=9,
                 sigma_color=75, sigma_space=50, window_center=None,
                 window_width=None):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.window_center = window_center
        self.window_width = window_width

    def apply_window_level(self, img, hu_min=-1000, hu_max=3000):
        if self.window_center is not None and self.window_width is not None:
            window_min = self.window_center - self.window_width // 2
            window_max = self.window_center + self.window_width // 2
            img = np.clip(img, window_min, window_max)
        img = ((img - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)
        return img

    def remove_noise(self, img, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return img

    def __call__(self, img, apply_denoise=True):
        if self.window_center is not None:
            img = self.apply_window_level(img)
        elif img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if len(img.shape) > 2 and img.shape[2] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if apply_denoise:
            img = cv2.fastNlMeansDenoising(img)

        img_clahe = self.clahe.apply(img)
        img_filtered = cv2.bilateralFilter(img_clahe, d=self.d,
                                           sigmaColor=self.sigma_color,
                                           sigmaSpace=self.sigma_space)

        return img_filtered


class CustomResNetModel(nn.Module):
    def __init__(self, input_shape=(1, 224, 224), num_classes=1):
        super(CustomResNetModel, self).__init__()
        self.base_model = resnet50(weights='DEFAULT')
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)

    def load_weights(self, path_to_weight):
        self.load_state_dict(torch.load(path_to_weight, map_location=torch.device('cpu')))
        print(f"Weights loaded from {path_to_weight}")


def classify(model, image_array, transform, device='cpu'):
    """Classify a single image using the given model and return the predicted class."""
    image = Image.fromarray(np.uint8(image_array)).convert('L')
    image = transform(image)
    image = image.unsqueeze(0)
    model.to(device)
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        prediction = torch.round(output)
    return prediction.item()


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    # Argument parsing
    parser = argparse.ArgumentParser(description="Classify CT images and save the results.")
    parser.add_argument("--patients_dir", type=str, required=True, help="Path to the directory containing patient data.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the model weights.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the classified dataset.")

    args = parser.parse_args()

    # Load model and weights
    model = CustomResNetModel()
    model.load_weights(args.weights_path)

    # Create the transform
    transform = CTTransform(
        clip_limit=2.0,
        tile_grid_size=(8, 8),
        d=9,
        sigma_color=75,
        sigma_space=50,
        window_center=40,
        window_width=400
    )

    classify_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load and process dataset
    dataset = {}
    pacients = os.listdir(args.patients_dir)

    for pacient in tqdm(pacients):
        dataset[pacient] = []
        slice_folder = os.path.join(args.patients_dir, pacient, "slice")

        for image in os.listdir(slice_folder):
            image_path = os.path.join(slice_folder, image)
            image_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image_data = transform(image_data)

            dataset[pacient].append({
                "image": image_path,
                "class": classify(model, image_data, classify_transform)
            })

    # Save dataset
    with open(args.output, 'w') as f:
        json.dump(dataset, f, indent=4)

    print(f"Classified dataset saved to {args.output}")
