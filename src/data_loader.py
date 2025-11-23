"""
Data loader for salient object detection.
Handles image preprocessing: resizing and normalization.
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms


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
                resized_image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)

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
                resized_mask = mask.resize((target_size, target_size), Image.Resampling.NEAREST)

                output_path = os.path.join(output_folder, filename)
                resized_mask.save(output_path)

                print(f"Resized: {filename}")
                successful += 1

            except Exception as e:
                print(f"Error with {filename}: {e}")
                failed += 1

    print(f"\nDone! Successful: {successful}, Failed: {failed}")


def normalize_image(image_path):
    """
    Load and normalize an image to [0, 1] range.

    Args:
        image_path: Path to image file

    Returns:
        Normalized numpy array with values in [0, 1]
    """
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image, dtype=np.float32)
    normalized = img_array / 255.0
    return normalized


def normalize_mask(mask_path):
    """
    Load and normalize a mask to [0, 1] range.

    Args:
        mask_path: Path to mask file

    Returns:
        Normalized numpy array with values in [0, 1]
    """
    mask = Image.open(mask_path).convert('L')
    mask_array = np.array(mask, dtype=np.float32)
    normalized = mask_array / 255.0
    return normalized


def apply_augmentation(image, mask, augment=True):
    """
    Apply augmentations to image and mask pair.
    Flip and crop are applied to both.
    Color transforms are only applied to image.

    Args:
        image: PIL Image (RGB)
        mask: PIL Image (grayscale)
        augment: If True, apply augmentations. If False, just convert to tensor.

    Returns:
        tuple: (image_tensor, mask_tensor)
    """
    if augment:
        # Random flip
        if np.random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        # Random crop (same params for both)
        if np.random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            )
            image = transforms.functional.resized_crop(image, i, j, h, w, image.size)
            mask = transforms.functional.resized_crop(mask, i, j, h, w, mask.size)

        # Color jitter (only for image, not mask)
        if np.random.random() > 0.5:
            image = transforms.ColorJitter(brightness=0.2, contrast=0.2)(image)

    # Convert to tensors and normalize
    image_tensor = transforms.ToTensor()(image)
    mask_tensor = transforms.ToTensor()(mask)

    return image_tensor, mask_tensor


def split_dataset(image_folder, mask_folder, test_size=0.30, random_state=42):
    """
    Split dataset into train (70%), validation (15%), and test (15%) sets.
    Uses sklearn's train_test_split.

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
        test_size=0.50,  # gjysa e 30% = 15% 
        random_state=random_state,
        shuffle=True
    )

    print(f"Dataset split: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")

    return train_pairs, val_pairs, test_pairs

if __name__ == '__main__':

    # Resize images
    resize_images(
        input_folder='data/ECSSD/images',
        output_folder='data/ECSSD/resized_images_128',
        target_size=128
    )

    # Resize masks
    resize_masks(
        input_folder='data/ECSSD/ground_truth_mask',
        output_folder='data/ECSSD/resized_masks_128',
        target_size=128
    )

    # Test normalization
    print("\nTesting normalization...")
    test_img = normalize_image('data/ECSSD/resized_images_128/0001.jpg')
    test_mask = normalize_mask('data/ECSSD/resized_masks_128/0001.png')

    print(f"Image shape: {test_img.shape}, range: [{test_img.min():.2f}, {test_img.max():.2f}]")
    print(f"Mask shape: {test_mask.shape}, range: [{test_mask.min():.2f}, {test_mask.max():.2f}]")

    # Test dataset split
    print("\nTesting dataset split...")
    train_pairs, val_pairs, test_pairs = split_dataset(
        image_folder='data/ECSSD/resized_images_128',
        mask_folder='data/ECSSD/resized_masks_128'
    )

    # Test augmentation
    print("\nTesting augmentation...")
    sample_image = Image.open('data/ECSSD/resized_images_128/0001.jpg')
    sample_mask = Image.open('data/ECSSD/resized_masks_128/0001.png')

    # With augmentation (training)
    aug_img, aug_mask = apply_augmentation(sample_image, sample_mask, augment=True)
    print(f"With augmentation - Image: {aug_img.shape}, Mask: {aug_mask.shape}")

    # Without augmentation (validation/test)
    img, mask = apply_augmentation(sample_image, sample_mask, augment=False)
    print(f"Without augmentation - Image: {img.shape}, Mask: {mask.shape}")

    print("\nPreprocessing complete.")
