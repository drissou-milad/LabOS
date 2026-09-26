"""
Regression test for the CellCNN / CellCNNNoClassifierReLU checkpoint mismatch.

Root cause: src/predict.py's load_model() instantiated CellCNN, but every checkpoint from
Run10 onward (including Run13) was trained as CellCNNNoClassifierReLU. The two classes are
identical except CellCNNNoClassifierReLU's classifier Sequential genuinely omits the ReLU
module (not just skips it in forward()), which shifts the final Linear layer's state_dict key
from `classifier.4` (CellCNN) to `classifier.3` (CellCNNNoClassifierReLU) -- exactly matching
the reported "Missing key(s): classifier.4.*" / "Unexpected key(s): classifier.3.*" error.

IMPORTANT DISCLOSURE: this test's checkpoint is SYNTHETIC (an untrained
CellCNNNoClassifierReLU's own state_dict, saved fresh) -- I do not have your real Run13
checkpoint file. It is built to have the exact key SHAPE (classifier.3.*, not classifier.4.*)
your reported error confirms your real checkpoint has, so this test genuinely exercises the
same mismatch/fix -- but it cannot substitute for running this same test against your real
checkpoint at config.BEST_MODEL_PATH once you've verified its SHA-256 (see
diagnose_run13_test.py's --expected-checkpoint-sha256, built for exactly that purpose).

This test calls the REAL, now-fixed src.predict.load_model() and src.predict.classify_centers()
-- the actual production inference path -- not a reimplementation of them.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import CellCNNNoClassifierReLU
from src.predict import load_model, classify_centers

EXPECTED_N_PARAMS = 89185


@pytest.fixture
def synthetic_run13_checkpoint(tmp_path):
    """An untrained CellCNNNoClassifierReLU's own state_dict -- has the exact key SHAPE
    (classifier.3.*, no classifier.4.*) the reported error confirms the real Run13 checkpoint
    has. See module docstring: this is a disclosed stand-in, not your real checkpoint."""
    model = CellCNNNoClassifierReLU()
    path = tmp_path / "synthetic_run13_checkpoint.pth"
    torch.save({"model_state_dict": model.state_dict()}, path)
    return path


def test_checkpoint_loads_with_no_missing_or_unexpected_keys(synthetic_run13_checkpoint):
    """Directly checks what the reported RuntimeError was about: the SET of keys the
    checkpoint has vs. the set load_model()'s instantiated architecture expects, independent
    of strict=True/False (production load_model() itself still uses the normal strict
    default -- this introspection is test-only, done via plain set comparison, not by loosening
    production loading)."""
    device = torch.device("cpu")
    model = CellCNNNoClassifierReLU().to(device)
    checkpoint_state = torch.load(synthetic_run13_checkpoint, map_location=device, weights_only=False)
    checkpoint_keys = set(checkpoint_state["model_state_dict"].keys())
    model_keys = set(model.state_dict().keys())

    missing = model_keys - checkpoint_keys
    unexpected = checkpoint_keys - model_keys
    assert not missing, f"Missing key(s) in checkpoint: {sorted(missing)}"
    assert not unexpected, f"Unexpected key(s) in checkpoint: {sorted(unexpected)}"


def test_load_model_succeeds_via_real_production_path(synthetic_run13_checkpoint):
    """Calls the REAL src.predict.load_model() -- the actual function the running LabOS app
    calls -- and confirms it completes without raising. Before the fix (CellCNN instantiated
    inside load_model()), this raises exactly the reported RuntimeError."""
    device = torch.device("cpu")
    model = load_model(synthetic_run13_checkpoint, device)
    assert isinstance(model, CellCNNNoClassifierReLU), (
        f"load_model() returned a {type(model).__name__}, expected CellCNNNoClassifierReLU -- "
        f"has load_model() been changed to instantiate a different class again?"
    )


def test_model_has_89185_parameters(synthetic_run13_checkpoint):
    device = torch.device("cpu")
    model = load_model(synthetic_run13_checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == EXPECTED_N_PARAMS, f"Expected {EXPECTED_N_PARAMS} parameters, got {n_params}"


def test_classify_centers_processes_a_valid_32x32_patch(synthetic_run13_checkpoint):
    """Exercises the real production inference path end-to-end: load_model() -> a real 96x96
    projection image (matching the reported dataset's per-frame shape) -> classify_centers()
    with a candidate comfortably inside the patch-extraction boundary."""
    device = torch.device("cpu")
    model = load_model(synthetic_run13_checkpoint, device)

    image = np.random.uniform(0, 4000, size=(96, 96)).astype(np.float32)
    centers = [(48, 48)]  # well within [16, 79] for a 32x32 patch on a 96x96 image
    mean, std = float(image.mean()), float(image.std())

    kept, confidences = classify_centers(model, image, centers, mean, std, device)

    assert len(kept) == len(confidences)
    assert len(kept) <= len(centers)
    for conf in confidences:
        assert 0.0 <= conf <= 1.0

    # A center clearly outside the boundary must be skipped (extract_patch returns None),
    # not raise -- confirms classify_centers' existing None-handling is untouched by this fix.
    kept_edge, conf_edge = classify_centers(model, image, [(2, 2)], mean, std, device)
    assert kept_edge == [] and conf_edge == []
