import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from models.dataset import PancreasDataset, SegmentationTransform  # Custom dataset for pancreas segmentation
from models.nest_unet import NestedUNet 
from models.utils import train_and_evaluate
from models.losses import BinaryDiceBCELoss

def main():

    parser = argparse.ArgumentParser(description="Train Att Nested U-Net for pancreas segmentation")

    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--split_ratio', type=float, default=0.8, help="Train/validation split ratio")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device for training")
    parser.add_argument('--deep_supervision', type=bool, default=False, help="Enable deep supervision")
    parser.add_argument('--save_model', type=bool, default=False, help="Save model")
    parser.add_argument('--model_path', type=str, default='./checkpoint', help="Save the best model")
    parser.add_argument('--model_name', type=str, default='nested_unet_pancreas.pth', help="Save the best model")
    parser.add_argument('--num_patients', type=int, default=40, help="Number of patients in training")
    parser.add_argument('--patience', type=int, default=None, help="Number of epochs without improvement in training")
    
    args = parser.parse_args()

    print('=========================================================')
    print("""
        ______  ______   ______   __   __   __                                      
        /\__  _\/\  == \ /\  __ \ /\ \ /\ "-.\ \                                     
        \/_/\ \/\ \  __< \ \  __ \\ \ \\ \ \-.  \                                    
           \ \_\ \ \_\ \_\\ \_\ \_\\ \_\\ \_\\"\_\                                   
            \/_/  \/_/ /_/ \/_/\/_/ \/_/ \/_/ \/_/                                   
         ______   ______  ______  ______   __   __   ______  __   ______   __   __   
        /\  __ \ /\__  _\/\__  _\/\  ___\ /\ "-.\ \ /\__  _\/\ \ /\  __ \ /\ "-.\ \  
        \ \  __ \\/_/\ \/\/_/\ \/\ \  __\ \ \ \-.  \\/_/\ \/\ \ \\ \ \/\ \\ \ \-.  \ 
         \ \_\ \_\  \ \_\   \ \_\ \ \_____\\ \_\\"\_\  \ \_\ \ \_\\ \_____\\ \_\\"\_\\
          \/_/\/_/   \/_/    \/_/  \/_____/ \/_/ \/_/   \/_/  \/_/ \/_____/ \/_/ \/_/
         __   __   ______   ______   ______  ______   _____                          
        /\ "-.\ \ /\  ___\ /\  ___\ /\__  _\/\  ___\ /\  __-.                        
        \ \ \-.  \\ \  __\ \ \___  \\/_/\ \/\ \  __\ \ \ \/\ \                       
         \ \_\\"\_\\ \_____\\/\_____\  \ \_\ \ \_____\\ \____-                       
          \/_/ \/_/ \/_____/ \/_____/   \/_/  \/_____/ \/____/                       
         __  __   __   __   ______  ______                                           
        /\ \/\ \ /\ "-.\ \ /\  ___\/\__  _\                                          
        \ \ \_\ \\ \ \-.  \\ \  __\\/_/\ \/                                          
         \ \_____\\ \_\\"\_\\ \_____\ \ \_\                                          
          \/_____/ \/_/ \/_/ \/_____/  \/_/                                          
""")

    # Data transformations
    transform = SegmentationTransform()

    # Load datasets
    dataset = PancreasDataset(
        data_dir=args.data_dir, 
        num_patients=args.num_patients,
        transform=transform
    )

    train_size = int(len(dataset) * args.split_ratio)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    for batch_idx, (images, targets) in enumerate(train_loader):
        print("Batch Example:")
        print("------------------------------")
        print(f"Images shape: {images.shape}")
        print(f"Targets shape: {targets.shape}")
        break  # Remove this to iterate through all batches

    # # Model, loss, optimizer
    model = NestedUNet(
        num_classes=1, 
        input_channels=1, 
        deep_supervision=args.deep_supervision
    ).to(args.device)

    criterion = BinaryDiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.00)

    # Training the model
    train_and_evaluate(
        model=model, 
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        epochs=args.epochs,
        save_model=args.save_model,
        model_path=args.model_path,
        model_name=args.model_name, 
        patience=args.patience
    )

if __name__ == "__main__":
    main()
