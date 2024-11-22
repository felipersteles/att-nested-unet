import os
import cv2
import numpy as np
import argparse
import math
from tqdm import tqdm
from glob import glob

# Função para carregar o atlas
def load_atlas(atlas_path):
    atlas = cv2.imread(atlas_path, cv2.IMREAD_GRAYSCALE)
    if atlas is None:
        raise FileNotFoundError(f"Atlas não encontrado no caminho: {atlas_path}")
    return atlas

# Função para encontrar o maior componente do atlas

def find_largest_component(atlas):
    binarized_atlas = (atlas > 0).astype(np.uint8) * 255  # Binariza o atlas
    contours, _ = cv2.findContours(binarized_atlas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Ajustar para múltiplos de 16
        w = math.ceil(w / 16) * 16
        h = math.ceil(h / 16) * 16

        return x, y, w, h
    else:
        print("Nenhum contorno encontrado!")
        return None

# Função para cortar imagens em pastas especificadas
def crop_images(patients_dir, output_dir, x, y, w, h):
    # Listar todos os diretórios de pacientes
    patient_dirs = sorted([d for d in os.listdir(patients_dir) if os.path.isdir(os.path.join(patients_dir, d))])

    for patient_dir in tqdm(patient_dirs):
        # Definindo os caminhos de entrada para as imagens
        pancreas_path = os.path.join(patients_dir, patient_dir, "pancreas", "*.jpg")
        slice_path = os.path.join(patients_dir, patient_dir, "slice", "*.jpg")

        # Criar diretórios de saída para o paciente
        output_patient_dir = os.path.join(output_dir, patient_dir)

        os.makedirs(os.path.join(output_patient_dir, "pancreas"), exist_ok=True)
        os.makedirs(os.path.join(output_patient_dir, "slice"), exist_ok=True)

        # Processar as imagens de pancreas
        for img_path in glob(pancreas_path):
            img = cv2.imread(img_path)
            if img is not None:
                img_cropped = img[y:y+h, x:x+w]
                # Criar um nome exclusivo para a imagem cortada
                base_name = os.path.basename(img_path)
                output_mask_path = os.path.join(output_patient_dir, "pancreas", f"{base_name.split('.')[0]}_cropped.jpg")
                cv2.imwrite(output_mask_path, img_cropped)

        # Processar as imagens de slice
        for img_path in glob(slice_path):
            img = cv2.imread(img_path)
            if img is not None:
                img_cropped = img[y:y+h, x:x+w]
                # Criar um nome exclusivo para a imagem cortada
                base_name = os.path.basename(img_path)
                output_slice_path = os.path.join(output_patient_dir, "slice", f"{base_name.split('.')[0]}_cropped.jpg")
                cv2.imwrite(output_slice_path, img_cropped)

if __name__ == "__main__":
    # Argument parser for command-line interface
    parser = argparse.ArgumentParser(description="Crop images based on a probabilistic atlas.")
    parser.add_argument("--atlas_path", type=str, required=True, help="Path to the atlas image file.")
    parser.add_argument("--patients_dir", type=str, required=True, help="Directory containing patient subdirectories.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save cropped images.")

    args = parser.parse_args()

    # Load the atlas and find the bounding box of the largest component
    atlas = load_atlas(args.atlas_path)
    bbox = find_largest_component(atlas)

    if bbox:
        x, y, w, h = bbox
        print(f"Largest component bounding box: x={x}, y={y}, w={w}, h={h}")

        # Crop images and save them to the output directory
        crop_images(args.patients_dir, args.output, x, y, w, h)
        print(f"Cropped images saved in: {args.output}")
    else:
        print("No valid bounding box found in the atlas.")
