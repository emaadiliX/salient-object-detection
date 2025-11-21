"""
Data loader for salient object detection.
Handles image preprocessing: resizing and normalization.
"""

import os
import numpy as np
from PIL import Image


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
                resized_image = image.resize((target_size, target_size), Image.LANCZOS)

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
                resized_mask = mask.resize((target_size, target_size), Image.NEAREST)

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
    print("\nPreprocessing complete.")
