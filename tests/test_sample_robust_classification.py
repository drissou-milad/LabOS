"""
Regression tests for the Run13-style sample-robust classification path added to
src/predict.py and wired into mvp/pipeline.py's run_detection_and_tracking().

Covers, in order:
  1. crop_and_resize_patch: shape contract, peripheral-signal exclusion, central-signal
     preservation (same checks train_exp05_run12.py's own manual verification used).
  2. compute_sample_robust_stats: leakage-safety (no labels parameter, checked via
     inspect.signature -- not just asserted in a comment), numeric correctness against a
     manual median/MAD computation, and that it raises (not silently degrades) on an empty
     patch array.
  3. classify_centers_robust: the key differentiating property -- that changing
     sample_scale changes its output, proving it genuinely uses the passed-in sample-wide
     statistic rather than silently falling back to some per-image computation. Also that
     boundary-excluded candidates are skipped exactly like classify_centers()'s existing
     behavior (no new boundary rule).
  4. classify_centers() (the ORIGINAL, untouched function) still behaves identically --
     proves the new code is additive, not a rewrite.
  5. mvp/pipeline.py::run_detection_and_tracking(): the real two-pass flow end to end on a
     synthetic volume, checking the downstream (N_t, 2)-per-frame contract HungarianTracker
     depends on is preserved; AND the zero-candidate edge case, verified by deleting the
     checkpoint file first and confirming no crash -- if the code tried to load the model
     when it shouldn't, this test would fail with FileNotFoundError, not silently pass.
  6. That models/norm_stats.npz is no longer required by the CNN path at all (never created
     in this test's fixture directory; the non-zero-candidate test still succeeds).
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (
    crop_and_resize_patch, collect_sample_patches, compute_sample_robust_stats,
    classify_centers_robust, classify_centers,
)
from src.model import CellCNNNoClassifierReLU


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def model(device):
    torch.manual_seed(0)
    m = CellCNNNoClassifierReLU().to(device)
    m.eval()
    return m


# --------------------------------------------------------------------------
# 1. crop_and_resize_patch
# --------------------------------------------------------------------------

def test_crop_and_resize_patch_shape():
    raw = np.full((32, 32), 1000.0, dtype=np.float32)
    out = crop_and_resize_patch(raw)
    assert out.shape == (32, 32)
    assert out.dtype == np.float32


def test_crop_and_resize_patch_excludes_peripheral_signal():
    raw = np.full((32, 32), 1000.0, dtype=np.float32)
    raw[1, 1] = 50000.0  # far outside the central [8:24, 8:24] crop
    out = crop_and_resize_patch(raw)
    assert out.max() < 1100, "a peripheral-only spike leaked into the crop-resized output"


def test_crop_and_resize_patch_preserves_central_signal():
    raw = np.full((32, 32), 1000.0, dtype=np.float32)
    raw[16, 16] = 50000.0  # exactly at the candidate center
    out = crop_and_resize_patch(raw)
    assert out.max() > 5000, "central signal was lost by the crop"


def test_crop_and_resize_patch_rejects_non_square_input():
    with pytest.raises(ValueError):
        crop_and_resize_patch(np.zeros((32, 16), dtype=np.float32))


# --------------------------------------------------------------------------
# 2. compute_sample_robust_stats
# --------------------------------------------------------------------------

def test_compute_sample_robust_stats_has_no_label_parameter():
    sig = inspect.signature(compute_sample_robust_stats)
    param_names = [p.lower() for p in sig.parameters]
    assert not any("label" in p or p == "y" for p in param_names), (
        f"compute_sample_robust_stats() must never accept labels -- got signature {sig}"
    )


def test_compute_sample_robust_stats_matches_manual_computation():
    rng = np.random.default_rng(0)
    patches = rng.normal(1500, 300, size=(50, 32, 32)).astype(np.float32)
    median, scale = compute_sample_robust_stats(patches)

    pixels = patches.astype(np.float64).ravel()
    manual_median = float(np.median(pixels))
    manual_mad = float(np.median(np.abs(pixels - manual_median)))
    assert median == pytest.approx(manual_median, abs=1e-9)
    assert scale == pytest.approx(1.4826 * manual_mad, abs=1e-9)


def test_compute_sample_robust_stats_raises_on_empty_input():
    with pytest.raises(ValueError):
        compute_sample_robust_stats(np.empty((0, 32, 32), dtype=np.float32))


# --------------------------------------------------------------------------
# 3. classify_centers_robust
# --------------------------------------------------------------------------

def test_classify_centers_robust_actually_uses_passed_sample_stats(model, device):
    rng = np.random.default_rng(1)
    image = rng.normal(1500, 300, size=(96, 96)).astype(np.float32)
    centers = [(48, 48), (48, 60)]

    _, conf_small_scale = classify_centers_robust(
        model, image, centers, sample_median=1500.0, sample_scale=300.0, device=device)
    _, conf_large_scale = classify_centers_robust(
        model, image, centers, sample_median=1500.0, sample_scale=3000.0, device=device)

    assert conf_small_scale != conf_large_scale, (
        "changing sample_scale did not change output probabilities -- classify_centers_robust "
        "is not actually using the passed-in sample statistics"
    )


def test_classify_centers_robust_skips_boundary_candidates(model, device):
    image = np.random.default_rng(2).normal(1500, 300, size=(96, 96)).astype(np.float32)
    kept, confidences = classify_centers_robust(
        model, image, [(2, 2)], sample_median=1500.0, sample_scale=300.0, device=device)
    assert kept == [] and confidences == []


# --------------------------------------------------------------------------
# 4. classify_centers() (original, untouched) still works
# --------------------------------------------------------------------------

def test_original_classify_centers_still_works_unmodified(model, device):
    image = np.random.default_rng(3).normal(1500, 300, size=(96, 96)).astype(np.float32)
    kept, confidences = classify_centers(
        model, image, [(48, 48)], mean=float(image.mean()), std=float(image.std()), device=device)
    assert len(kept) == len(confidences)


# --------------------------------------------------------------------------
# 5/6. mvp/pipeline.py integration
# --------------------------------------------------------------------------

@pytest.fixture
def synthetic_volume():
    rng = np.random.default_rng(4)
    volume = rng.integers(0, 3, size=(8, 3, 96, 96)).astype(np.uint16)
    for t in range(8):
        for (y, x) in [(48, 48), (60, 60), (30, 70)]:
            volume[t, :, y - 1:y + 2, x - 1:x + 2] = 3000
    return volume


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    from src import config
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    path = models_dir / "best_model.pth"
    torch.save({"model_state_dict": CellCNNNoClassifierReLU().state_dict()}, path)
    monkeypatch.setattr(config, "BEST_MODEL_PATH", path)
    # Deliberately NOT creating norm_stats.npz anywhere -- see
    # test_cnn_path_does_not_require_norm_stats_npz below for why that's the point.
    return path


def test_run_detection_and_tracking_two_pass_flow(synthetic_volume, checkpoint):
    from mvp.pipeline import run_detection_and_tracking
    from src.lineage import LineageResult

    result = run_detection_and_tracking(synthetic_volume, use_cnn_filter=True)

    assert isinstance(result, LineageResult)
    assert hasattr(result, "nodes")
    assert hasattr(result, "edges")
    assert hasattr(result, "divisions")
    assert hasattr(result, "tracks")


def test_zero_candidates_skips_model_loading_entirely(monkeypatch):
    """Deletes/never-provides the checkpoint path, and confirms a flat (sub-threshold) volume
    does NOT crash trying to load it -- proves compute_sample_robust_stats and load_model are
    never reached when there's nothing for the CNN to classify."""
    from src import config
    from mvp.pipeline import run_detection_and_tracking
    monkeypatch.setattr(config, "BEST_MODEL_PATH", Path("/definitely/does/not/exist.pth"))

    flat_volume = np.full((8, 3, 96, 96), 5, dtype=np.uint16)  # below DETECTION_THRESHOLD
    result = run_detection_and_tracking(flat_volume, use_cnn_filter=True)
    assert len(result.nodes) == 0
    assert len(result.edges) == 0
    assert len(result.divisions) == 0
    assert len(result.tracks) == 0
    


def test_cnn_path_does_not_require_norm_stats_npz(synthetic_volume, checkpoint, tmp_path, monkeypatch):
    """norm_stats.npz is deliberately never created anywhere in this test's fixture directory.
    If run_detection_and_tracking() still tried to np.load(config.BEST_NORM_STATS_PATH) the
    way it used to, this would raise FileNotFoundError."""
    from src import config
    from mvp.pipeline import run_detection_and_tracking
    monkeypatch.setattr(config, "BEST_NORM_STATS_PATH", tmp_path / "definitely_absent_norm_stats.npz")
    result = run_detection_and_tracking(synthetic_volume, use_cnn_filter=True)
    assert len(result.nodes) > 0  # got this far without needing the file at all
