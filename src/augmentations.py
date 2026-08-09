"""
Lightweight augmentations for normalized 2D grayscale patches (numpy arrays).

These operate on already-normalized (zero-mean, unit-ish scale) patches, as
produced in notebooks/04_training.ipynb cell 13. Apply them to TRAINING
patches only — never to validation/test patches.
"""

import numpy as np

from src import config


def random_flip(patch):
    if np.random.rand() < 0.5:
        patch = np.fliplr(patch)
    if np.random.rand() < 0.5:
        patch = np.flipud(patch)
    return patch


def random_rotation(patch):
    """Random rotation by a multiple of 90 degrees (safe for square patches,
    no interpolation artifacts)."""
    k = np.random.randint(0, 4)
    return np.rot90(patch, k)


def random_brightness(patch, max_delta=None):
    """Additive brightness jitter. Since patches are already standardized,
    this shifts the mean by a small fraction of a standard deviation."""
    max_delta = config.BRIGHTNESS_MAX_DELTA if max_delta is None else max_delta
    delta = np.random.uniform(-max_delta, max_delta)
    return patch + delta


def gaussian_noise(patch, sigma=None):
    sigma = config.GAUSSIAN_NOISE_SIGMA if sigma is None else sigma
    noise = np.random.normal(0.0, sigma, size=patch.shape).astype(patch.dtype)
    return patch + noise


def augment_patch(patch):
    """
    Apply the standard augmentation stack to a single normalized patch.

    `np.fliplr` / `np.flipud` / `np.rot90` return views with negative
    strides, which `torch.tensor(...)` cannot consume directly — the
    `.copy()` at the end is required, not cosmetic.
    """
    patch = random_flip(patch)
    patch = random_rotation(patch)
    patch = random_brightness(patch)
    patch = gaussian_noise(patch)
    return patch.copy()
