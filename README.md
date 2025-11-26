# Salient Object Detection

A CNN-based salient object detection system that identifies and segments the most visually prominent objects in images.

## Dataset Setup

1. Download the ECSSD dataset from: https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html
2. Create the following folder structure:
   ```
   data/
   └── ECSSD/
       ├── images/              <- Put original images here
       └── ground_truth_mask/   <- Put ground truth masks here
   ```
3. Run the data loader to resize images to 128x128:
   ```
   cd src
   python data_loader.py
   ```
   This creates `resized_images_128/` and `resized_masks_128/` folders.

## Project Structure

### src/

| File             | Description                                                                                  |
| ---------------- | -------------------------------------------------------------------------------------------- |
| `data_loader.py` | Handles image resizing, dataset splitting (70/15/15), augmentation, and DataLoader creation. |
| `sod_model.py`   | Defines the encoder-decoder CNN architecture                                                 |
| `train.py`       | Training loop with validation, early stopping, LR scheduler, and checkpoint resume.          |
| `evaluate.py`    | Evaluates the model on test set and generates visualization samples.                         |

### Running the Code

1. **Train the model:**

   ```
   cd src
   python train.py
   ```

   You will see epoch-by-epoch progress with Train Loss, Train IoU, Val Loss, Val IoU, and learning rate. Training stops early if validation loss doesn't improve for 5 epochs.

2. **Evaluate the model:**
   ```
   cd src
   python evaluate.py
   ```
   This prints IoU, Precision, Recall, F1-Score, and MAE metrics on the test set, and saves sample visualizations to the `visualizations/` folder.

### Output Files

| Path              | Description                                                                      |
| ----------------- | -------------------------------------------------------------------------------- |
| `best_model.pth`  | Saved model weights from the best validation epoch.                              |
| `visualizations/` | Contains sample images showing input, ground truth, predicted mask, and overlay. |

## Demo Notebook

`demo.ipynb` is a simple demonstration for running inference on a single image. It loads the model architecture from `sod_model.py` and the trained weights from `best_model.pth`, then displays:

- Input image
- Predicted saliency mask
- Overlay visualization
- Inference time in milliseconds

Change the `image_path` variable to test different images.

## Google Colab Version

`SOD_TRAINING_COLAB.ipynb` contains a Google Colab demo where the same code was run with a larger dataset, better GPU, more batches, more epochs, and higher resolution (256x256) for improved results with more computing power.
