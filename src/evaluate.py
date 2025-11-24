"""
Evaluation script for salient object detection model.
"""
import os
import torch
from sod_model import SODModel
from data_loader import split_dataset, create_dataloaders


def calculate_metrics(pred, target, threshold=0.5):
    """Calculate IoU, Precision, Recall, F1-Score, and MAE"""
    # Convert predictions to binary (0 or 1)
    pred_binary = (pred > threshold).float()

    # Target masks are already binary (0 or 1)
    target_binary = target

    # True Positives, False Positives, False Negatives
    tp = (pred_binary * target_binary).sum()
    fp = (pred_binary * (1 - target_binary)).sum()
    fn = ((1 - pred_binary) * target_binary).sum()

    # IoU
    intersection = tp
    union = tp + fp + fn
    iou = (intersection + 1e-6) / (union + 1e-6)

    # Precision, Recall, F1
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    # MAE (Mean Absolute Error)
    mae = (pred - target).abs().mean()

    return iou.item(), precision.item(), recall.item(), f1.item(), mae.item()


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in test_loader:
            # Move to device
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate metrics
            iou, precision, recall, f1, mae = calculate_metrics(outputs, masks)
            total_iou += iou
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_mae += mae
            num_batches += 1

    # Average metrics across all batches
    avg_iou = total_iou / num_batches
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches
    avg_f1 = total_f1 / num_batches
    avg_mae = total_mae / num_batches

    return avg_iou, avg_precision, avg_recall, avg_f1, avg_mae


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_pairs, val_pairs, test_pairs = split_dataset(
        image_folder='data/ECSSD/resized_images_128',
        mask_folder='data/ECSSD/resized_masks_128'
    )

    _, _, test_loader = create_dataloaders(
        train_pairs, val_pairs, test_pairs, batch_size=16
    )

    # Load model
    model_path = 'best_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Train the model first.")
        return

    model = SODModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # Evaluate
    avg_iou, avg_precision, avg_recall, avg_f1, avg_mae = evaluate_model(model, test_loader, device)

    print("\nTest Set Evaluation Results")
    print("="*50)
    print(f"IoU (Intersection over Union): {avg_iou:.4f}")
    print(f"Precision:                     {avg_precision:.4f}")
    print(f"Recall:                        {avg_recall:.4f}")
    print(f"F1-Score:                      {avg_f1:.4f}")
    print(f"Mean Absolute Error:           {avg_mae:.4f}")


if __name__ == '__main__':
    main()
