import os 
import nibabel as nib
import numpy as np
import cv2

def save_nifti_with_masks(input, output, file_name):
    
    images = f'{input}/images/{file_name}'
    masks = f'{input}/labels/{file_name}'

    # # Create directories to save the frames and masks
    patient = file_name.split(".")[0]
    output_directory = f'{output}/{patient}/slice'
    mask_output_directory_1 = f'{output}/{patient}/pancreas'
    mask_output_directory_2 = f'{output}/{patient}/cancer'

    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(mask_output_directory_1, exist_ok=True)
    os.makedirs(mask_output_directory_2, exist_ok=True)

    # # Load NIfTI data
    image_nii_data = nib.load(images)
    mask_nii_data = nib.load(masks)

    # # Get the data arrays
    image_data = image_nii_data.get_fdata()
    mask_data = mask_nii_data.get_fdata()

    # # Define Hounsfield windowing
    window_center = 75  # Center of the Hounsfield window
    window_width = 400  # Width of the Hounsfield window
    lower_bound = window_center - window_width // 2  # -150
    upper_bound = window_center + window_width // 2  # 250

    # Create frames with the image and mask overlay
    for i in range(image_data.shape[2]):  # Loop over each slice (assumes axial slices)
        image_frame = image_data[:, :, i]
        mask_frame = mask_data[:, :, i]

        # Apply Hounsfield windowing
        image_frame = np.clip(image_frame, lower_bound, upper_bound)
        image_frame = (image_frame - lower_bound) / (upper_bound - lower_bound) * 255  # Normalize to 0-255
        image_frame = image_frame.astype(np.uint8)

        # Create color masks for visualization (assuming two masks)
        color_mask_1 = np.zeros((*mask_frame.shape, 3), dtype=np.uint8)
        color_mask_2 = np.zeros((*mask_frame.shape, 3), dtype=np.uint8)

        # Binary masks: 1 for pancreas, 2 for cancer
        color_mask_1[mask_frame == 1] = [255, 0, 0]  # Red for pancreas
        color_mask_2[mask_frame == 2] = [0, 255, 0]  # Green for cancer

        # Combine mask1 and mask2 for saving as mask1
        combined_mask_1 = (mask_frame == 1) | (mask_frame == 2)  # Combine regions of mask1 and mask2
        combined_mask_1 = combined_mask_1.astype(np.uint8)  # Convert to uint8 (0 or 1)

        color_mask_2[mask_frame == 2] = [255, 255, 255]

        # Save the combined frames as JPG images
        cv2.imwrite(os.path.join(output_directory, f'{patient}_{i:03d}.jpg'), image_frame)
        cv2.imwrite(os.path.join(mask_output_directory_1, f'{patient}_{i:03d}.jpg'), combined_mask_1 * 255)  # Save combined mask as binary
        cv2.imwrite(os.path.join(mask_output_directory_2, f'{patient}_{i:03d}.jpg'), color_mask_2)

