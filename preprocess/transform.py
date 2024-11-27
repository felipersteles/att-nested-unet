import argparse
import os
from tqdm import tqdm
from models.utils import save_nifti_with_masks

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert a 3D image to 2D slices.")

    parser.add_argument("--input", type=str, required=True, help="Path to the 3D image file (e.g., .npy file).")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the 2D images.")
    parser.add_argument("--axis", type=int, default=2, help="Axis along which to slice (0, 1, or 2).")
    parser.add_argument("--index", type=int, default=None, help="Specific index of the slice to extract.")

    args = parser.parse_args()

    # Load the 3D image

    data_folder = "images"
    
    file_names = os.listdir(args.input + "/" + data_folder)

    for file_name in tqdm(file_names):
        save_nifti_with_masks(
            file_name=file_name,
            input=args.input,
            output=args.output
        )

    print("Processing complete.")
