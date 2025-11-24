"""
Training script for salient object detection model.
"""
from sod_model import SODModel, get_optimizer, train_model
from data_loader import split_dataset, create_dataloaders


def main():
    # Split dataset
    train_pairs, val_pairs, test_pairs = split_dataset(
        image_folder='data/ECSSD/resized_images_128',
        mask_folder='data/ECSSD/resized_masks_128'
    )

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_pairs, val_pairs, test_pairs, batch_size=16
    )

    # Create model
    model = SODModel()

    # Create optimizer
    optimizer = get_optimizer(model, learning_rate=1e-3)

    # Train model
    print("\nStarting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        epochs=25,
        patience=5
    )

    print("\nTraining complete! Best model saved to: best_model.pth")


if __name__ == '__main__':
    main()
