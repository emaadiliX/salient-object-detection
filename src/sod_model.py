"""
CNN model for salient object detection.
"""
import torch
import torch.nn as nn


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


if __name__ == '__main__':
    model = SODModel()

    # Test input (2 images, 3 channels, 128x128)
    test_input = torch.randn(2, 3, 128, 128)
    print(f"Input shape: {test_input.shape}")

    output = model(test_input)
    print(f"Output shape: {output.shape}")

    print("\nModel works!")
