from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

from src import config
from src.augmentations import augment_patch


def extract_patches(projection, centers, patch_size=None, exclusion_radius=None):
    """
    Extract a positive patch at each detected center, and an equal number of
    random negative patches kept away from every center.

    Shared by src/train.py and notebooks/04_training_demo.ipynb so patch
    extraction logic exists in exactly one place.
    """
    patch_size = patch_size or config.PATCH_SIZE
    exclusion_radius = exclusion_radius or config.NEGATIVE_EXCLUSION_RADIUS
    half = patch_size // 2

    positive_patches = []
    for y, x in centers:
        y1, y2 = y - half, y + half
        x1, x2 = x - half, x + half
        if y1 < 0 or x1 < 0 or y2 >= projection.shape[0] or x2 >= projection.shape[1]:
            continue
        positive_patches.append(projection[y1:y2, x1:x2])

    negative_patches = []
    attempts = 0
    # Bounded retry budget — the original notebook's `while` loop had no cap
    # and could hang forever on a dense/small image where every random point
    # lands within exclusion_radius of a center.
    max_attempts = max(len(positive_patches), 1) * 200 + 1000

    while len(negative_patches) < len(positive_patches) and attempts < max_attempts:
        attempts += 1

        y = np.random.randint(half, projection.shape[0] - half)
        x = np.random.randint(half, projection.shape[1] - half)

        too_close = any(
            np.sqrt((cy - y) ** 2 + (cx - x) ** 2) < exclusion_radius
            for cy, cx in centers
        )
        if too_close:
            continue

        patch = projection[y - half:y + half, x - half:x + half]
        if patch.shape == (patch_size, patch_size):
            negative_patches.append(patch)

    if len(negative_patches) < len(positive_patches):
        raise RuntimeError(
            f"Only found {len(negative_patches)}/{len(positive_patches)} negative "
            f"patches after {max_attempts} attempts. Image may be too small/dense "
            f"for the current PATCH_SIZE / NEGATIVE_EXCLUSION_RADIUS."
        )

    return positive_patches, negative_patches


def build_patch_arrays(positive_patches, negative_patches, seed=None):
    """Stack patch lists into shuffled (X, y) numpy arrays."""
    X = np.array(positive_patches + negative_patches).astype(np.float32)
    y = np.concatenate([
        np.ones(len(positive_patches), dtype=np.float32),
        np.zeros(len(negative_patches), dtype=np.float32),
    ])

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


class PatchDataset(Dataset):
    """
    Torch Dataset over pre-extracted, pre-normalized 2D patches
    (see src/train.py or notebooks/04_training_demo.ipynb for the
    normalization step).

    Augmentation is applied on-the-fly per __getitem__ call, so training
    and validation should use SEPARATE PatchDataset instances:
    `PatchDataset(X_train, y_train, augment=True)` and
    `PatchDataset(X_val, y_val, augment=False)`.
    """

    def __init__(self, X, y, augment=False):
        self.X = X.numpy() if torch.is_tensor(X) else np.asarray(X)
        self.y = y.numpy() if torch.is_tensor(y) else np.asarray(y)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        patch = self.X[idx]

        # Accept both (H, W) and (1, H, W) input shapes.
        if patch.ndim == 3:
            patch = patch[0]

        if self.augment:
            patch = augment_patch(patch)

        patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.y[idx], dtype=torch.float32)

        return patch, label


class BioHubDataset:
    """
    Dataset loader for the BioHub Cell Tracking competition.
    """

    def __init__(self, root):
        self.root = Path(root)

        self.train_dir = self.root / "train"
        self.test_dir = self.root / "test"

        if not self.train_dir.exists():
            raise FileNotFoundError(f"Train folder not found: {self.train_dir}")

    # --------------------------------------------------
    # Sample lists
    # --------------------------------------------------

    @property
    def train_samples(self):
        return sorted([p.stem for p in self.train_dir.glob("*.zarr")])

    @property
    def test_samples(self):
        return sorted([p.stem for p in self.test_dir.glob("*.zarr")])

    # --------------------------------------------------
    # Loading data
    # --------------------------------------------------

    def load_volume(self, sample_name, split="train"):

        folder = self.train_dir if split == "train" else self.test_dir

        volume_path = folder / f"{sample_name}.zarr"

        return zarr.open(volume_path, mode="r")["0"]

    def load_graph(self, sample_name):

        graph_path = self.train_dir / f"{sample_name}.geff"

        return zarr.open_group(graph_path, mode="r")

    # --------------------------------------------------
    # Information
    # --------------------------------------------------

    def info(self):

        print("=" * 60)
        print("BioHub Dataset")
        print("=" * 60)

        print(f"Root        : {self.root}")
        print(f"Train files : {len(self.train_samples)}")
        print(f"Test files  : {len(self.test_samples)}")