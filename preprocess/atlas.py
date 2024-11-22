import os
import cv2
import numpy as np
from glob import glob

# Função para carregar todas as máscaras existentes
def load_masks(patients_dir):
    masks = []

    # Listar todos os diretórios de pacientes presentes
    patient_dirs = sorted([d for d in os.listdir(patients_dir) if os.path.isdir(os.path.join(patients_dir, d))])
    print(f"Number of patient directories: {len(patient_dirs)}")

    for patient_dir in patient_dirs:
        # Caminho para as máscaras de cada paciente
        mask_paths = sorted(glob(os.path.join(patients_dir, patient_dir, "pancreas", "*.jpg")))

        for mask_path in mask_paths:
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:  # Verificar se a imagem foi carregada corretamente
                masks.append(mask_img)

    print(f"Total number of masks loaded: {len(masks)}")
    return masks

# Função para criar o atlas probabilístico
def create_probabilistic_atlas(masks):
    atlas = np.zeros_like(masks[0], dtype=np.float32)

    for mask in masks:
        atlas += mask / 255.0  # Normaliza as máscaras para [0, 1]

    atlas /= len(masks)  # Média das máscaras
    return atlas

# Função para encontrar o maior componente do atlas
def find_largest_component(atlas):
    # Binarizar o atlas
    binarized_atlas = (atlas > 0).astype(np.uint8) * 255  # Threshold adaptado para pegar a região relevante

    # Encontrar contornos no atlas binarizado
    contours, _ = cv2.findContours(binarized_atlas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Identificar o maior contorno
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, w, h
    else:
        print("Nenhum contorno encontrado!")
        return None

# Função para salvar o atlas com bounding box
def save_atlas_with_bounding_box(atlas, x, y, w, h, output_path):
    # Converte o atlas para o formato apropriado para salvamento
    atlas_normalized = (atlas * 255).astype(np.uint8)  # Normaliza para [0, 255]

    # Desenhar a bounding box
    if x is not None and y is not None:
        cv2.rectangle(atlas_normalized, (x, y), (x + w, y + h), (255, 255, 255), 2)  # Desenha a bounding box em branco

    os.makedirs(output_path, exist_ok=True)

    # Salvar a imagem
    cv2.imwrite(output_path + "/atlas.jpg", atlas_normalized)
    print(f"Atlas saved with bounding box at {output_path}")

if __name__ == "__main__":
    import argparse

    # Configuração do argparse
    parser = argparse.ArgumentParser(description="Create a probabilistic atlas from segmentation masks.")
    parser.add_argument("--input", type=str, required=True, help="Path to the directory containing patient subdirectories.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the resulting atlas with bounding box.")
    args = parser.parse_args()

    # Carregar máscaras
    print("Loading masks...")
    masks = load_masks(args.input)

    # Criar atlas probabilístico
    print("Creating probabilistic atlas...")
    atlas = create_probabilistic_atlas(masks)

    # Encontrar maior componente
    print("Finding largest component in the atlas...")
    bounding_box = find_largest_component(atlas)
    if bounding_box:
        x, y, w, h = bounding_box
        print(f"Bounding box found: x={x}, y={y}, w={w}, h={h}")
    else:
        x, y, w, h = None, None, None, None
        print("No bounding box found!")

    # Salvar atlas com bounding box
    print("Saving atlas with bounding box...")
    save_atlas_with_bounding_box(atlas, x, y, w, h, args.output)

    print("Process complete.")
