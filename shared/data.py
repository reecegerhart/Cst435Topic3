"""Fashion-MNIST data helpers shared by training (Colab) and the API (Render).

Importing this module needs only numpy + Pillow. torchvision is imported lazily
inside `load_fashion_mnist`, so the API tier never has to install it.

Splits: the official 60k training set is split into train/val; the official 10k
test set is NEVER used for training or model selection.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

IMG_SIZE = 28
CHANNELS = 1
CLASS_NAMES: List[str] = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
N_CLASSES = len(CLASS_NAMES)


def load_fashion_mnist(root: str = "data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download (first run only) and return uint8 arrays.

    Returns ``(x_train [60000,28,28], y_train, x_test [10000,28,28], y_test)``.
    """
    from torchvision import datasets  # lazy: training-only dependency

    train = datasets.FashionMNIST(root=root, train=True, download=True)
    test = datasets.FashionMNIST(root=root, train=False, download=True)
    return (
        train.data.numpy().astype(np.uint8),
        train.targets.numpy().astype(np.int64),
        test.data.numpy().astype(np.uint8),
        test.targets.numpy().astype(np.int64),
    )


def make_splits(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Deterministic train/val split of the official training set.

    Returns ``{"train": (x, y), "val": (x, y), "test": (x, y)}``.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x_train))
    n_val = int(len(x_train) * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return {
        "train": (x_train[train_idx], y_train[train_idx]),
        "val": (x_train[val_idx], y_train[val_idx]),
        "test": (x_test, y_test),
    }


def to_gray_28(img: Image.Image) -> np.ndarray:
    """Convert any PIL image to the model's input format: [28, 28] uint8 grayscale.

    Used by the API for uploads. Note: Fashion-MNIST items are light-on-dark;
    an arbitrary photo is not, which is part of the AI-literacy lesson.
    """
    return np.asarray(img.convert("L").resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8)