import argparse
import itertools
import torch
from torch.utils.data import DataLoader, random_split
from models.dataset import PancreasDataset, SegmentationTransform
from models.losses import BinaryDiceBCELoss
from models.nest_unet import NestedUNet
from models.utils import train_and_evaluate

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Grid search for hyperparameter tuning on Pancreas Segmentation")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset")
    parser.add_argument('--num_patients', type=int, default=None, help="Number of patients to include in the dataset")
    parser.add_argument('--split_ratio', type=float, default=0.8, help="Train/val split ratio (default: 0.8)")
    parser.add_argument('--save_dir', type=str, default="./grid_search_results", help="Directory to save results and models")
    args = parser.parse_args()

    # Define the hyperparameter grid
    param_grid = {
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "batch_size": [8, 16, 32],
        "weight_decay": [0.0, 1e-5],
    }

    # Create combinations of all hyperparameters
    param_combinations = list(itertools.product(
        param_grid["learning_rate"],
        param_grid["batch_size"],
        param_grid["weight_decay"],
    ))

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations
    transform = SegmentationTransform()

    # Load datasets
    dataset = PancreasDataset(
        data_dir=args.data_dir, 
        num_patients=args.num_patients,
        transform=transform
    )

    # Split dataset into train and validation sets
    train_size = int(len(dataset) * args.split_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Initialize results storage
    results = []

    # Iterate over hyperparameter combinations
    for i, (lr, batch_size, weight_decay) in enumerate(param_combinations):
        print('=========================================================')
        print('=========================================================')
        print(f"Starting run {i + 1}/{len(param_combinations)} with params:")
        print(f"Learning rate: {lr}, Batch size: {batch_size}, Weight decay: {weight_decay}")
        print('=========================================================')
        print('=========================================================')

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize the model
        model = NestedUNet(
            num_classes=1, 
            input_channels=1, 
            deep_supervision=False
        ).to(device)

        # Initialize the optimizer and criterion
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = BinaryDiceBCELoss()

        # Train and evaluate the model
        val_loss = train_and_evaluate(
            model=model,
            epochs=10,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            patience=None,
            save_model=False,
            model_path=args.save_dir,
            model_name=f"model_run_{i + 1}.pth"
        )

        # Store the results
        results.append({
            "params": {
                "learning_rate": lr,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
            },
            "val_loss": val_loss
        })

    # Find the best result
    best_result = min(results, key=lambda x: x["val_loss"])
    print("Best Result:")
    print(f"Params: {best_result['params']}, Validation Loss: {best_result['val_loss']}")

    # Save results to a file
    with open(f"{args.save_dir}/grid_search_results.txt", "w") as f:
        for result in results:
            f.write(f"Params: {result['params']}, Validation Loss: {result['val_loss']}\n")
        f.write("\nBest Result:\n")
        f.write(f"Params: {best_result['params']}, Validation Loss: {best_result['val_loss']}\n")

if __name__ == "__main__":
    main()
