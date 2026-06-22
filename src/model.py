"""A small CRNN for handwritten text recognition.

The network is the classic convolutional recurrent design used for sequence
recognition. Convolutions shrink the image height to one and turn the width into
a time axis, then a bidirectional GRU reads that sequence and a linear head emits
per timestep class scores for CTC.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes: int, image_height: int = 32, hidden_size: int = 64):
        super().__init__()
        if image_height % 4 != 0:
            raise ValueError("image_height must be divisible by 4")
        self.num_classes = num_classes
        self.image_height = image_height

        # Two stride 2 height reductions take 32 -> 16 -> 8 in height. A final
        # pooling over the remaining height collapses it to 1 in the forward pass.
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2, W/2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/4, W/4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.cnn_out_channels = 128
        self.rnn = nn.GRU(
            input_size=self.cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
        )
        self.head = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of images to per timestep log probabilities.

        Args:
            x: tensor shaped (batch, 1, height, width).

        Returns:
            log_probs: tensor shaped (time, batch, num_classes), ready to be fed
            straight into nn.CTCLoss or the greedy decoder.
        """
        feats = self.conv(x)  # (B, C, H', W')
        # Collapse the remaining height into the channel statistics so each
        # width column becomes one feature vector.
        feats = feats.mean(dim=2)  # (B, C, W')
        feats = feats.permute(2, 0, 1)  # (W', B, C) == (T, B, C)
        seq, _ = self.rnn(feats)  # (T, B, 2*hidden)
        logits = self.head(seq)  # (T, B, num_classes)
        return logits.log_softmax(dim=-1)
