"""
CNN model for salient object detection.
"""
import torch
import torch.nn as nn
import torch.optim as optim


class SODModel(nn.Module):
    def __init__(self):
        super(SODModel, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.relu_up1 = nn.ReLU()

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.relu_up2 = nn.ReLU()

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.relu_up3 = nn.ReLU()

        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.relu_up4 = nn.ReLU()

        # Output
        self.output_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        # Decoder
        x = self.upconv1(x)
        x = self.relu_up1(x)

        x = self.upconv2(x)
        x = self.relu_up2(x)

        x = self.upconv3(x)
        x = self.relu_up3(x)

        x = self.upconv4(x)
        x = self.relu_up4(x)

        # Output
        x = self.output_conv(x)
        x = self.sigmoid(x)

        return x


def sod_loss(pred, target):
    """Calculate loss"""
    bce = nn.BCELoss()
    bce_loss = bce(pred, target)

    # IoU
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)

    loss = bce_loss + 0.5 * (1 - iou)

    return loss


def get_optimizer(model, learning_rate=1e-3):
    return optim.Adam(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    model = SODModel()
    optimizer = get_optimizer(model, learning_rate=1e-3)

    # Test input (2 images, 3 channels, 128x128)
    test_input = torch.randn(2, 3, 128, 128)
    print(f"Input shape: {test_input.shape}")

    output = model(test_input)
    print(f"Output shape: {output.shape}")

    print(f"Optimizer: Adam with lr={optimizer.param_groups[0]['lr']}")

    print("\nModel works!")
