"""
Data loader for salient object detection.
Handles image preprocessing, dataset creation, and data loading.
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import platform


def resize_images(input_folder, output_folder, target_size=224):
    """Resize images using Lanczos interpolation."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = os.listdir(input_folder)
    successful = 0
    failed = 0

    for filename in image_files:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            try:
                input_path = os.path.join(input_folder, filename)
                image = Image.open(input_path)
                resized_image = image.resize(
                    (target_size, target_size), Image.Resampling.LANCZOS)

                output_path = os.path.join(output_folder, filename)
                resized_image.save(output_path)

                print(f"Resized: {filename}")
                successful += 1

            except Exception as e:
                print(f"Error with {filename}: {e}")
                failed += 1

    print(f"\nDone! Successful: {successful}, Failed: {failed}")


def resize_masks(input_folder, output_folder, target_size=224):
    """Resize masks using nearest-neighbor to preserve binary values."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mask_files = os.listdir(input_folder)
    successful = 0
    failed = 0

    for filename in mask_files:
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            try:
                input_path = os.path.join(input_folder, filename)
                mask = Image.open(input_path)
                resized_mask = mask.resize(
                    (target_size, target_size), Image.Resampling.NEAREST)

                output_path = os.path.join(output_folder, filename)
                resized_mask.save(output_path)

                print(f"Resized: {filename}")
                successful += 1

            except Exception as e:
                print(f"Error with {filename}: {e}")
                failed += 1

    print(f"\nDone! Successful: {successful}, Failed: {failed}")


def split_dataset(image_folder, mask_folder, test_size=0.30, random_state=42):
    """
    Split dataset into train (70%), validation (15%), and test (15%) sets.

    Args:
        image_folder: Folder containing images
        mask_folder: Folder containing masks
        test_size: Proportion for val+test combined (default 0.30)
        random_state: Random seed for reproducibility

    Returns:
        tuple: (train_pairs, val_pairs, test_pairs)
    """
    # Collect all image-mask file pairs
    image_files = sorted([f for f in os.listdir(image_folder)
                         if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))])

    file_pairs = []
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        mask_file = os.path.splitext(img_file)[0] + '.png'
        mask_path = os.path.join(mask_folder, mask_file)

        if os.path.exists(mask_path):
            file_pairs.append((img_path, mask_path))

    # Split into train (70%) and temp (30%)
    train_pairs, temp_pairs = train_test_split(
        file_pairs,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # Split temp into val (15%) and test (15%)
    val_pairs, test_pairs = train_test_split(
        temp_pairs,
        test_size=0.50,
        random_state=random_state,
        shuffle=True
    )

    print(
        f"Dataset split: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")

    return train_pairs, val_pairs, test_pairs


def apply_augmentation(image, mask, augment=True):
    """
    Apply augmentations to image and mask pair, then convert to tensors.
    All images are normalized to [0, 1] range during tensor conversion.

    Args:
        image: PIL Image (RGB)
        mask: PIL Image (grayscale)
        augment: If True, apply augmentations. If False, just convert to tensor.

    Returns:
        tuple: (image_tensor, mask_tensor) both normalized to [0, 1]
    """
    if augment:
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        # Random crop
        if np.random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            )
            image = transforms.functional.resized_crop(
                image, i, j, h, w, image.size)
            mask = transforms.functional.resized_crop(
                mask, i, j, h, w, mask.size)

        # Brightness variation (only for image)
        if np.random.random() > 0.5:
            image = transforms.ColorJitter(brightness=0.2, contrast=0.2)(image)

    # Convert to tensors (automatically normalizes to 0 and 1)
    image_tensor = transforms.ToTensor()(image)
    mask_tensor = transforms.ToTensor()(mask)

    return image_tensor, mask_tensor


class SODDataset(Dataset):
    """
    PyTorch Dataset for Salient Object Detection.
    Loads image-mask pairs and applies augmentation.
    """

    def __init__(self, file_pairs, augment=False):
        """
        Args:
            file_pairs: List of (image_path, mask_path) tuples
            augment: Whether to apply data augmentation
        """
        self.file_pairs = file_pairs
        self.augment = augment

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        """
        Load and return a single image-mask pair.

        Returns:
            tuple: (image_tensor, mask_tensor)
        """
        img_path, mask_path = self.file_pairs[idx]

        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Apply augmentation and convert to tensors
        image_tensor, mask_tensor = apply_augmentation(
            image, mask, self.augment)

        return image_tensor, mask_tensor


def create_dataloaders(train_pairs, val_pairs, test_pairs, batch_size=16):
    """
    Create PyTorch DataLoaders for train, validation, and test sets.

    Args:
        train_pairs: List of training (image_path, mask_path) pairs
        val_pairs: List of validation pairs
        test_pairs: List of test pairs
        batch_size: Number of samples per batch
        num_workers: Number of subprocesses for data loading 

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    system = platform.system()
    if system == "Windows":
        num_workers = 0
    else:
        num_workers = 2

    print(f"Detected OS: {system} → num_workers={num_workers}")

    # Create datasets
    train_dataset = SODDataset(train_pairs, augment=True)
    val_dataset = SODDataset(val_pairs, augment=False)
    test_dataset = SODDataset(test_pairs, augment=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"DataLoaders created:")
    print(
        f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':

    # Resize images
    resize_images(
        input_folder='data/ECSSD/images',
        output_folder='data/ECSSD/resized_images_128',
        target_size=128)

    # Resize masks
    resize_masks(
        input_folder='data/ECSSD/ground_truth_mask',
        output_folder='data/ECSSD/resized_masks_128',
        target_size=128)

    # Split dataset
    print("\nChecking dataset split...")
    train_pairs, val_pairs, test_pairs = split_dataset(
        image_folder='data/ECSSD/resized_images_128',
        mask_folder='data/ECSSD/resized_masks_128')

    # Create dataloaders
    print("\nCreating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_pairs, val_pairs, test_pairs, batch_size=16)

    # Quick check on one batch
    print("\nChecking a batch from train_loader...")
    images, masks = next(iter(train_loader))
    print(
        f"  images shape: {images.shape}, range: [{images.min():.3f}, {images.max():.3f}]")
    print(
        f"  masks  shape: {masks.shape}, range: [{masks.min():.3f}, {masks.max():.3f}]")
    print(f"  image dtype: {images.dtype}, mask dtype: {masks.dtype}")

    # Simple augmentation check
    print("\nQuick augmentation check...")

    # pick one random sample from train set
    sample_img_path, sample_mask_path = train_pairs[0]
    img = Image.open(sample_img_path).convert('RGB')
    mask = Image.open(sample_mask_path).convert('L')

    # Apply augmentation
    aug_img, aug_mask = apply_augmentation(img, mask, augment=True)

    print(
        f"Augmented image shape: {aug_img.shape}, range [{aug_img.min():.3f}, {aug_img.max():.3f}]")
    print(
        f"Augmented mask  shape: {aug_mask.shape}, range [{aug_mask.min():.3f}, {aug_mask.max():.3f}]")

    print("\nData loading pipeline ready!")
