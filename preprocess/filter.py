import os
import shutil

def get_images_path(source, pacient, image_path):
    """Get the full paths for slice, pancreas, and cancer images."""
    image_name = image_path.split('/')[-1]

    image_slice = os.path.join(source, image_path.split('/')[-3], "slice", image_name)
    image_pancreas = os.path.join(source, image_path.split('/')[-3], "pancreas", image_name)
    image_cancer = os.path.join(source, image_path.split('/')[-3], "cancer", image_name)

    return image_slice, image_pancreas, image_cancer

def filter_dataset(dataset, origin_folder, destination_folder):
    """Filter the dataset and copy the relevant images to the destination folder."""
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for pacient in dataset.keys():
        for image in dataset[pacient]:
            if image['class'] == 1:  # Filter condition
                image_path = image['image']

                # Get paths for source images
                image_slice, image_pancreas, image_cancer = get_images_path(origin_folder, pacient, image_path)

                # Get paths for destination images
                slice_path, pancreas_path, cancer_path = get_images_path(destination_folder, pacient, image_path)

                # Create destination directories
                os.makedirs(os.path.dirname(slice_path), exist_ok=True)
                os.makedirs(os.path.dirname(pancreas_path), exist_ok=True)
                # Uncomment the line below if cancer images need to be copied
                # os.makedirs(os.path.dirname(cancer_path), exist_ok=True)

                # Copy files to destination
                shutil.copy(image_slice, slice_path)
                shutil.copy(image_pancreas, pancreas_path)
                # Uncomment the line below if cancer images need to be copied
                # shutil.copy(image_cancer, cancer_path)

if __name__ == "__main__":
    import argparse
    import json

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Filter dataset and copy relevant images to the destination folder.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSON file containing the dataset.")
    parser.add_argument("--origin_folder", type=str, required=True, help="Path to the source folder containing images.")
    parser.add_argument("--destination_folder", type=str, required=True, help="Path to the destination folder.")

    args = parser.parse_args()

    # Load the dataset from the JSON file
    with open(args.dataset_path, 'r') as f:
        dataset = json.load(f)

    # Filter the dataset and save to the destination folder
    filter_dataset(dataset, args.origin_folder, args.destination_folder)
    print(f"Filtered images saved in: {args.destination_folder}")
