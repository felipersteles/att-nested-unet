import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage
from models.metrics import precision_score, dice_coefficient, calc_jaccard_index
from PIL import Image

class CTTransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), d=5, sigma_color=75, sigma_space=25):
        # CLAHE parameters
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        self.d = d  # Diameter of each pixel neighborhood
        self.sigma_color = sigma_color  # Filter sigma in the color space
        self.sigma_space = sigma_space  # Filter sigma in the coordinate space

    def __call__(self, img):
        # Apply CLAHE to the image
        img_clahe = self.clahe.apply(img)

        # Apply Bilateral Filter
        img_filtered = cv2.bilateralFilter(img_clahe, d=self.d, sigmaColor=self.sigma_color, sigmaSpace=self.sigma_space)

        return img_filtered

def import_dataset(dir, num_patients=None):

    patients = os.listdir(dir)

    if num_patients is not None:
      if num_patients > len(patients):
        raise ValueError(f'Max number of pacients is {len(patients)}')
      else:
        patients = patients[:num_patients]

    dataset = {}
    for patients in tqdm(patients):
      dataset[patients] = {
          'images': [],
          'masks': []
      }

      slice_folder = os.path.join(dir, patients + '/slice')
      mask_folder = os.path.join(dir, patients + '/pancreas')

      for image in os.listdir(slice_folder):
        dataset[patients]['images'].append(os.path.join(slice_folder, image))

      for mask in os.listdir(mask_folder):
        dataset[patients]['masks'].append(os.path.join(mask_folder, mask))


    return dataset

def import_images(dataset):

    images, masks = [], []

    transform = CTTransform()

    print('=========================================================')
    with tqdm(total=len(dataset.keys()), desc='Loading data for patients') as patient_bar:
        for patients in dataset.keys():
            # Process images (slices) stored in .npy files
            for image_path in dataset[patients]['images']:
                # Load image from .npy file
                image_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Apply the transformation
                image_data = transform(image_data)
                images.append(image_data)

            # Process masks stored in .npy files
            for mask_path in dataset[patients]['masks']:
                # Load mask from .npy file (no transformation for mask)
                mask_data = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if len(mask_data.shape) == 3 and mask_data.shape[-1] == 3:
                    # If the mask has 3 channels (RGB), convert it to grayscale
                    mask_data = cv2.cvtColor(mask_data, cv2.COLOR_RGB2GRAY)

                masks.append(mask_data)

            patient_bar.update(1)

    images = np.array(images)
    masks = np.array(masks)

    print('=========================================================')
    return images, masks

def import_data_and_show_summary(data_dir, num_patients=None):
    
    dataset = import_dataset(data_dir, num_patients)
    images, masks = import_images(dataset)

    data_summary(images, masks, 'Train')

    return images, masks
    

def dataset_summary(dataset):
  print('=========================================================')
  print('Dataset summary:')
  print('------------------------------')
  print(f'Number of patient: {len(dataset.keys())}')
  print(f'Number of images: {sum([len(dataset[patient]["images"]) for patient in dataset.keys()])}')
  print(f'Number of masks: {sum([len(dataset[patient]["masks"]) for patient in dataset.keys()])}')
  print('=========================================================')

def data_summary(train_images, train_masks, title):

    pancreas_train = np.sum(np.any(train_masks > 0, axis=(1, 2)))

    # Show summary after import images
    print('=========================================================')
    print(f'{title} summary:')
    print('------------------------------')
    print(f'Number of images for {title}: {train_images.shape}')
    print(f'Number of masks for {title}: {train_masks.shape}')
    print('------------------------------')
    print(f'Images with pancreas: {pancreas_train}')
    print('=========================================================')


def train_epoch(model, show_epoch, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0

    train_loader_tqdm = tqdm(dataloader, desc=f"Training epoch {show_epoch}", leave=False)
    for images, masks in train_loader_tqdm:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def validate_epoch(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    all_labels = []
    all_preds = []

    val_loader_tqdm = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for images, labels in val_loader_tqdm:
            images = images.to(device)
            labels = labels.to(device).float()  # Ensure labels are float for loss calculation

            # Forward pass
            outputs = model(images)  # Expected shape: [batch_size, 1, H, W]

            # Compute loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Apply thresholding to obtain binary predictions
            predicted = (outputs >= 0.5).float()

            # Collect labels and predictions for metric calculation
            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(predicted.cpu().numpy().flatten())

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Ensure binary values for metrics (0 or 1)
    all_labels = (all_labels >= 0.5).astype(int)
    all_preds = (all_preds >= 0.5).astype(int)

    # Compute metrics
    avg_val_loss = val_loss / len(val_loader)
    dice_coeff = dice_coefficient(all_preds, all_labels)
    jaccard_index = calc_jaccard_index(all_labels, all_preds, zero_division=1)
    precision = precision_score(all_preds, all_labels)


    return avg_val_loss, dice_coeff, jaccard_index, precision

def train_and_evaluate(model, epochs, train_loader, val_loader, criterion, optimizer, device, patience=None, save_model=False, model_path='./checkpoint', model_name='best_model.pth'):

    print("=========================================================")
    print(" __                   _                          _  ")
    print("(_ _|_ _ __ _|_ o __ (_|   _|_ __ _  o __  o __ (_| ")
    print("__) |_(_|| ' |_ | | |__|    |_ | (_| | | | | | |__| ")
    print("=========================================================")
    print("Starting training for", epochs, "epochs")
    print("=========================================================")

    metrics = {
        'val_loss': [],
        'dice_coeff': [],
        'jaccard_index': [],
        'precision': []
    }
    
    best_val_loss = float('inf')
    best_dice =0

    torch.cuda.empty_cache()
    
    for epoch in range(epochs):
        show_epoch = epoch + 1

        train_loss = train_epoch(model, show_epoch, train_loader, criterion, optimizer, device)
        print(f"Epoch [{show_epoch}/{epochs}] Training Loss: {train_loss:.4f}")
        print('------------------------------')
        
        val_loss, dice_coeff, jaccard_index, precision  = validate_epoch(model, val_loader, criterion, device)

        metrics['val_loss'].append(val_loss)
        metrics['dice_coeff'].append(dice_coeff)
        metrics['jaccard_index'].append(jaccard_index)
        metrics['precision'].append(precision)

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Dice Coefficient: {dice_coeff:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Jaccard Index: {jaccard_index:.4f}")

        # Storage the best dice
        if dice_coeff > best_dice:
            best_dice = dice_coeff

            if save_model is True:
                if not os.path.exists(model_path):
                    os.makedirs(model_path)

                torch.save(model.state_dict(), model_path + "/" + model_name)
                print(f"Saved best model weights with Dice: {dice_coeff:.4f} at epoch: {show_epoch}")

        # Check if this is the best Val Loss and save weights if needed
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0 
        else:
            epochs_without_improvement += 1

        print("=========================================================")

        # Early stopping check based on Val Loss only if patience is set
        if patience is not None and epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {show_epoch} epochs without improvement.")
            break

    print(f"The best DICE achieved was {best_dice:.4f}.")
    return metrics

def visualize_segmentation(input_tensor, output_tensor, mask_tensor):
    """
    Visualizes segmentation results using the NestedUNetWithMultiheadAttention model.

    Args:
        input_tensor: Input image tensor of shape (1, C, H, W).
        output_tensor: Predicted segmentation tensor of shape (1, C, H, W).
        mask_tensor: Ground truth mask tensor of shape (1, C, H, W).
    """
    # Convert input image, predicted output, and ground truth mask to displayable formats
    input_image_display = input_tensor.squeeze(0).cpu().numpy()
    if input_image_display.shape[0] == 1:  # Grayscale image
        input_image_display = input_image_display[0]

    mask_tensor_display = mask_tensor.squeeze(0).cpu().numpy()
    if mask_tensor_display.shape[0] == 1:  # Grayscale mask
        mask_tensor_display = mask_tensor_display[0]

    # Plot input image, predicted mask, and ground truth mask
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the input image
    ax[0].imshow(input_image_display, cmap='gray')
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    # Plot the predicted segmentation
    ax[1].imshow(output_tensor, cmap='gray')
    ax[1].set_title('Predicted Segmentation')
    ax[1].axis('off')

    # Plot the ground truth mask
    ax[2].imshow(mask_tensor_display, cmap='gray')
    ax[2].set_title('Ground Truth Mask')
    ax[2].axis('off')

    # Tight layout and display
    plt.tight_layout()
    plt.show()

def compare_model_state_dicts(model, weights_path, device='cpu'):
    """
    Compare the state_dict keys of the current model with the loaded state_dict.

    Args:
        model (torch.nn.Module): The model to compare.
        weights_path (str): Path to the saved weights.
        device (str): The device to load the weights on ('cpu' or 'cuda').
    """
    # Load saved state_dict
    saved_state_dict = torch.load(weights_path, map_location=device, weights_only=True)

    # Get state_dict keys
    saved_keys = set(saved_state_dict.keys())
    model_keys = set(model.state_dict().keys())

    # Print saved state_dict keys
    # print("Keys in Saved State Dict:")
    # for key in saved_keys:
    #     print(key)
    # print("--------------------------------")

    # # Print model state_dict keys
    # print("Keys in Model State Dict:")
    # for key in model_keys:
    #     print(key)
    # print("--------------------------------")

    # Compare keys
    missing_keys = model_keys - saved_keys
    extra_keys = saved_keys - model_keys

    if not missing_keys and not extra_keys:
        print("The state_dict keys match perfectly!")
    else:
        if missing_keys:
            print("--------------------------------")
            print("Missing keys:", len(missing_keys))
            print("--------------------------------")
            print("Missing keys in saved state_dict:")
            for key in missing_keys:
                print(key)

        if extra_keys:
            print("--------------------------------")
            print("Extra keys:", len(extra_keys))
            print("--------------------------------")
            print("Extra keys in saved state_dict:")
            for key in extra_keys:
                print(key)
