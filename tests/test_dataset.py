import numpy as np
import torch

from src.dataset import extract_patches, build_patch_arrays, PatchDataset


def make_synthetic_projection(size=128, centers=((30, 30), (60, 60), (90, 40))):
    """A blank image with bright blobs at known centers, so extract_patches
    has real, well-separated positive locations to work with."""
    image = np.zeros((size, size), dtype=np.float32)
    for y, x in centers:
        image[y - 2:y + 2, x - 2:x + 2] = 5000.0
    return image, np.array(centers)


def test_extract_patches_counts_and_shape():
    image, centers = make_synthetic_projection()
    positive, negative = extract_patches(image, centers, patch_size=16, exclusion_radius=10)

    assert len(positive) == len(centers)
    assert len(negative) == len(positive)
    for patch in positive + negative:
        assert patch.shape == (16, 16)


def test_extract_patches_negatives_are_far_from_centers():
    image, centers = make_synthetic_projection()
    exclusion_radius = 15
    _, negative = extract_patches(image, centers, patch_size=16, exclusion_radius=exclusion_radius)

    # We don't have the negative centers directly (only patches), but we can
    # at least confirm none of the negative patches contain a bright blob
    # pixel (5000.0), which would indicate the exclusion radius failed.
    for patch in negative:
        assert patch.max() < 5000.0


def test_extract_patches_drops_out_of_bounds_centers():
    image, _ = make_synthetic_projection(size=64)
    # A center right at the edge can't have a full patch extracted around it
    edge_centers = np.array([[1, 1], [30, 30]])
    positive, _ = extract_patches(image, edge_centers, patch_size=16, exclusion_radius=5)

    assert len(positive) == 1  # only the in-bounds center survives


def test_build_patch_arrays_labels_and_shuffle():
    positives = [np.ones((8, 8), dtype=np.float32) for _ in range(5)]
    negatives = [np.zeros((8, 8), dtype=np.float32) for _ in range(5)]

    X, y = build_patch_arrays(positives, negatives, seed=0)

    assert X.shape == (10, 8, 8)
    assert y.sum() == 5
    assert set(np.unique(y)) == {0.0, 1.0}


def test_patch_dataset_len_and_item_shape():
    X = np.random.randn(6, 32, 32).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)

    ds = PatchDataset(X, y, augment=False)
    assert len(ds) == 6

    patch, label = ds[0]
    assert isinstance(patch, torch.Tensor)
    assert patch.shape == (1, 32, 32)
    assert label.shape == ()


def test_patch_dataset_augmentation_preserves_shape():
    X = np.random.randn(4, 32, 32).astype(np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.float32)

    ds = PatchDataset(X, y, augment=True)
    for i in range(len(ds)):
        patch, _ = ds[i]
        assert patch.shape == (1, 32, 32)
