import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from models.nest_unet import NestedUNet
from models.utils import visualize_segmentation, compare_model_state_dicts
from models.metrics import calc_jaccard_index, dice_coefficient
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image


def load_image_and_mask(image_path, mask_path, transform):
    # Load and preprocess the input image
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Load and preprocess the mask
    mask = Image.open(mask_path).convert("L")  # Convert to grayscale
    mask_tensor = transform(mask).unsqueeze(0)  # Add batch dimension

    return image_tensor, mask_tensor

def main():
    # Define model parameters
    num_classes = 1
    input_channels = 1
    deep_supervision = False

    # Initialize model
    model = NestedUNet(num_classes=num_classes, input_channels=input_channels, deep_supervision=deep_supervision)

    # Path to pretrained weights
    weights_path = "./weights/nested_unet_pancreas_4.pth"  # Replace with the actual path

    # Compare weights
    compare_model_state_dicts(model, weights_path, device='cpu')

    # Define image transformations
    input_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    

    # Directory containing images and masks
    image_dir = "data/cropped_filtered/test/efec/slice"
    mask_dir = "data/cropped_filtered/test/efec/pancreas"

    # Iterate over all images in the directory
    for image_name in os.listdir(image_dir):
        # Construct paths to the image and mask
        image_path = os.path.join(image_dir, image_name)
        mask_path = os.path.join(mask_dir, image_name)  # Assuming mask has the same name as image

        # Load the image and mask
        input_tensor, mask_tensor = load_image_and_mask(image_path, mask_path, input_transform)

        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(input_tensor.to(device))
        
        # Assuming deep supervision is not used, take the last output
        output = output[-1] if isinstance(output, list) else output
        output = output.squeeze().cpu().numpy()  # Apply sigmoid to get probabilities

        mask = mask_tensor.squeeze().cpu().numpy()

        # Convert the output and mask to binary (0 or 1)
        output_bin = (output > 0.5).astype(np.uint8)
        mask_bin = (mask > 0.5).astype(np.uint8)

        # Calculate Dice coefficient and Jaccard index
        dice = dice_coefficient(output_bin, mask_bin)
        jaccard = calc_jaccard_index(output_bin.flatten(), mask_bin.flatten())

        print("------------------------------------------------")
        print(f"Output min: {output.min()}, max: {output.max()}")
        print(f"Image: {image_name}")
        print(f"Dice Coefficient: {dice:.4f}")
        print(f"Jaccard Index: {jaccard:.4f}")

        plt.imshow(output, cmap='gray')
        plt.title('Raw Output (Sigmoid Applied)')
        plt.colorbar()
        plt.show()

        # Visualize segmentation result (optional)
        visualize_segmentation(input_tensor, output_bin, mask_tensor)

if __name__ == "__main__":
    main()

    print("------------------------------------------------")
