"""
Simple CNN model for salient object detection.
"""
import torch
import torch.nn as nn


class SODModel(nn.Module):
    def __init__(self):
        super(SODModel, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()  # Activation function (adds non-linearity)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 3
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 4
        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder Layer 1
        x = self.conv1(x)   # Shape: (batch, 64, 128, 128)
        x = self.relu1(x)
        x = self.pool1(x)   # (batch, 64, 64, 64)

        # Encoder Layer 2
        x = self.conv2(x)   # Shape: (batch, 128, 64, 64)
        x = self.relu2(x)
        x = self.pool2(x)   # (batch, 128, 32, 32)

        # Encoder Layer 3
        x = self.conv3(x)   # Shape: (batch, 256, 32, 32)
        x = self.relu3(x)
        x = self.pool3(x)   # (batch, 256, 16, 16)

        # Encoder Layer 4
        x = self.conv4(x)   # Shape: (batch, 512, 16, 16)
        x = self.relu4(x)
        x = self.pool4(x)   # (batch, 512, 8, 8)

        return x
