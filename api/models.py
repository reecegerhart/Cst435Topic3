"""CNN architectures. Imported by the training script (Colab) AND the API (Render).

Both models expose `cam_layer`: the module whose output Grad-CAM hooks into
(the last conv block's activations, before the final pooling, so the heatmap
is 14x14 instead of a coarse 7x7).
"""
from __future__ import annotations

from torch import nn

from shared.data import CHANNELS, N_CLASSES


class BaselineCNN(nn.Module):
    """Two conv blocks + fully connected head. No regularization tricks."""

    def __init__(self, n_classes: int = N_CLASSES, channels: int = CHANNELS):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 0,1,2 -> 14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),        # 3,4,5 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    @property
    def cam_layer(self) -> nn.Module:
        return self.features[4]  # ReLU after 2nd conv, 14x14

    def forward(self, x):
        return self.classifier(self.features(x))


class ImprovedCNN(nn.Module):
    """Same widths as the baseline, plus batch norm and dropout.

    (Augmentation, weight decay, and the LR schedule live in the training script.)
    """

    def __init__(self, n_classes: int = N_CLASSES, channels: int = CHANNELS, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),  # 0-3
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),        # 4-7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    @property
    def cam_layer(self) -> nn.Module:
        return self.features[6]  # ReLU after 2nd conv + BN, 14x14

    def forward(self, x):
        return self.classifier(self.features(x))


MODEL_KINDS = {"baseline": BaselineCNN, "improved": ImprovedCNN}


def build_model(kind: str) -> nn.Module:
    if kind not in MODEL_KINDS:
        raise ValueError(f"unknown model kind {kind!r}; choose from {list(MODEL_KINDS)}")
    return MODEL_KINDS[kind]()