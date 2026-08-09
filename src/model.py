import torch.nn as nn


class CellCNN(nn.Module):
    """
    Patch-level binary classifier: is the center of this 32x32 patch a cell?

    v2 architecture: 3 conv blocks (up from 2) with BatchNorm for training
    stability and Dropout before the final layer for regularization on a
    small dataset. Returns raw LOGITS (no Sigmoid) — pair with
    nn.BCEWithLogitsLoss() or src.losses.FocalLoss(), both of which expect
    logits. Apply torch.sigmoid() only at inference time.
    """

    def __init__(self, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 16x16 -> 8x8
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 8x8 -> 4x4
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            # No Sigmoid — see class docstring.
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
