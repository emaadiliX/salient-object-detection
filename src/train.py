"""
Training and validation loop with checkpoint resume functionality.
"""
import os
import torch
from sod_model import SODModel, sod_loss, get_optimizer
from data_loader import split_dataset, create_dataloaders


def train_model(model, train_loader, val_loader, optimizer, epochs=25, patience=5, checkpoint_path='checkpoint.pth'):
    """Train model with early stopping and checkpoint resume"""
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    best_val_loss = 999999.0
    no_improve_count = 0
    start_epoch = 0

    # Resume from checkpoint if exists
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        no_improve_count = checkpoint['no_improve_count']
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        total_train_loss = 0.0
        total_train_iou = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = sod_loss(outputs, masks)

            # Calculate IoU metric
            intersection = (outputs * masks).sum()
            union = outputs.sum() + masks.sum() - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            total_train_iou += iou.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_iou = total_train_iou / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        total_val_iou = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                # Move to device
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = sod_loss(outputs, masks)
                total_val_loss += loss.item()

                # Calculate IoU metric
                intersection = (outputs * masks).sum()
                union = outputs.sum() + masks.sum() - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                total_val_iou += iou.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_iou = total_val_iou / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}, Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_count = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("  Saved best model")
        else:
            no_improve_count += 1
            print(f"  No improvement ({no_improve_count}/{patience})")

        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'no_improve_count': no_improve_count
        }
        torch.save(checkpoint, checkpoint_path)

        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Remove checkpoint after successful training
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Removed checkpoint file")

    print(f"Training done. Best val loss: {best_val_loss:.4f}")


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
