"""Standalone inference script.

Usage:
    python -m src.predict --sample <sample_name> --split test --out outputs/predictions.csv
    python -m src.predict --sample <sample_name> --frame 40 --checkpoint models/best_model.pth

Loads a trained CellCNNNoClassifierReLU checkpoint, runs the existing
CellDetector across the complete sample volume, computes Run13 sample-wide
robust normalization statistics from all candidate patches, classifies the
requested frame, and writes accepted centers with their confidence to a CSV.

The detector and tracker implementations are not modified by this script.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom as _scipy_zoom

from src import config
from src.dataset import BioHubDataset
from src.detector import CellDetector
from src.logging_utils import setup_logging
from src.model import CellCNNNoClassifierReLU

logger = setup_logging()

# --------------------------------------------------------------------------
# Run13-style sample-robust classification path (crop+resize field-of-view change +
# per-sample robust normalization).
#
# The legacy classify_centers() path is retained for backward compatibility, while production
# inference in main() now uses the Run13 two-pass flow below. The existing extract_patch()
# transform remains unchanged.
#
# Exactly reproduces Run13's raw-domain transform (crop_and_resize_patch, inherited unchanged
# from Run12) and Run13's per-sample robust normalization protocol (compute_sample_robust_stats
# / classify_centers_robust), as trained by scripts/benchmark_v2/train_exp05_run13.py and
# evaluated by scripts/benchmark_v2/diagnose_run13_test.py -- see those files for the original
# derivation and the measured ~0.75px centering-offset note this crop/resize carries.
#
# WHY classify_centers_robust() remains separate from classify_centers():
# Run13-style normalization needs median/MAD computed over an entire SAMPLE's candidate patches
# pooled together (see compute_sample_robust_stats) -- not a single image's own mean/std the
# way classify_centers() uses. That statistic must exist BEFORE any single candidate can be
# classified, so the caller (mvp/pipeline.py) needs a genuinely two-pass flow: collect every
# candidate patch across the whole volume first, compute stats once, then classify. Bolting
# that onto classify_centers() would change its signature and its single-image calling
# contract; keeping it separate means the original per-patch-instance-normalization path
# (used by main() below, and by any historical/regression code that still calls
# classify_centers() directly) keeps working unmodified.

ROBUST_SCALE_CONSTANT = 1.4826  # matches Run11/12/13's own constant exactly
SAMPLE_NORM_EPS = 1e-6          # matches Run11/12/13's NORM_EPS_FLOOR: a pure numerical-
                                 # stability guard, not a tuned noise-floor-fraction -- see
                                 # train_exp05_run11.py's own normalization_config.json note
                                 # for why that concept doesn't transfer to a population-level
                                 # statistic like this one
CROP_HALF = 8                   # patch[16-8:16+8, 16-8:16+8] = patch[8:24, 8:24]
RESIZE_ORDER = 1                # bilinear, matches train_exp05_run12.py/run13.py exactly


def crop_and_resize_patch(raw_patch, crop_half=CROP_HALF, resize_order=RESIZE_ORDER):
    """Exactly reproduces Run12/Run13's field-of-view transform: crop the central
    (2*crop_half, 2*crop_half) region out of a config.PATCH_SIZE x config.PATCH_SIZE raw patch
    (as returned by extract_patch(), candidate at local index (16,16) by construction), then
    resize back to config.PATCH_SIZE x config.PATCH_SIZE via scipy.ndimage.zoom. Deterministic
    (no RNG) for a fixed resize_order. Raises ValueError if raw_patch isn't square or if the
    resize doesn't land on an exact integer output size, exactly as train_exp05_run12.py does,
    rather than silently producing a mismatched shape."""
    size = raw_patch.shape[0]
    if raw_patch.shape != (size, size):
        raise ValueError(f"crop_and_resize_patch expects a square input, got {raw_patch.shape}.")
    center = size // 2
    crop = raw_patch[center - crop_half: center + crop_half, center - crop_half: center + crop_half]
    if crop.shape != (2 * crop_half, 2 * crop_half):
        raise ValueError(f"Center crop has shape {crop.shape}, expected "
                          f"({2*crop_half},{2*crop_half}).")
    zoom_factor = size / (2 * crop_half)
    resized = _scipy_zoom(crop.astype(np.float64), zoom=zoom_factor, order=resize_order)
    if resized.shape != (size, size):
        raise ValueError(f"Resized patch has shape {resized.shape}, expected ({size},{size}) -- "
                          f"zoom_factor={zoom_factor} did not land on an exact integer output size.")
    return resized.astype(np.float32)


def collect_sample_patches(images_per_frame, centers_per_frame, patch_size=None):
    """Pass 1 of the two-pass Run13-style flow: extracts (via the existing, unmodified
    extract_patch()) and crop-resizes every candidate's raw patch across every frame of the
    CURRENT sample/volume, pooling them into one array. Candidates too close to the image
    boundary for a full patch are skipped exactly as extract_patch() already does elsewhere --
    not a new boundary rule. Returns an (N, patch_size, patch_size) float32 array (N may be 0,
    in which case the caller must not proceed to compute_sample_robust_stats -- see its own
    docstring). Label-independent by construction: this function's signature has no labels
    parameter, and images_per_frame/centers_per_frame come from the detector, not from any
    ground truth."""
    patch_size = patch_size or config.PATCH_SIZE
    patches = []
    for image, centers in zip(images_per_frame, centers_per_frame):
        for y, x in centers:
            patch = extract_patch(image, y, x, patch_size)
            if patch is None:
                continue
            patches.append(crop_and_resize_patch(patch))
    if not patches:
        return np.empty((0, patch_size, patch_size), dtype=np.float32)
    return np.stack(patches).astype(np.float32)


def compute_sample_robust_stats(patches):
    """Pass 2 of the two-pass Run13-style flow: median/MAD over the POOLED RAW PIXELS of every
    patch in `patches` (as produced by collect_sample_patches -- already crop-resized, NOT yet
    normalized). Exactly Run13's own compute_per_sample_robust_stats formula, minus the
    sample_id grouping (production inference classifies one volume/experiment at a time, so
    there is only ever one "sample" here). Label-independent by construction: this function's
    signature has no labels parameter and never reads one -- it is a pure function of the pixel
    population, matching the leakage-safety property established for Run11/12/13's own training-
    time statistics.

    Raises ValueError on an empty array rather than returning a degenerate (nan/inf) stat --
    the caller (mvp/pipeline.py) must check for zero total candidates BEFORE calling this, not
    rely on this function to fail gracefully for that case."""
    if patches.shape[0] == 0:
        raise ValueError("compute_sample_robust_stats() called with zero patches -- the caller "
                          "must check for zero candidates across the whole volume before calling "
                          "this (see mvp/pipeline.py's run_detection_and_tracking()).")
    pixels = patches.astype(np.float64).ravel()
    median = float(np.median(pixels))
    mad = float(np.median(np.abs(pixels - median)))
    scale = ROBUST_SCALE_CONSTANT * mad
    return median, scale


@torch.no_grad()
def classify_centers_robust(model, image, centers, sample_median, sample_scale, device,
                             eps=SAMPLE_NORM_EPS, patch_size=None):
    """Run13-style classification for one frame's candidates, using PRECOMPUTED sample-wide
    statistics (from compute_sample_robust_stats over the WHOLE volume's candidates, not this
    frame's own mean/std -- that is the entire point of this function existing separately from
    classify_centers()). Same per-frame calling shape as classify_centers() (one image, its
    centers, device) and the same (kept_centers, confidences) return contract and 0.5
    probability threshold -- only the preprocessing differs:

        raw patch (extract_patch, unchanged)
            -> crop_and_resize_patch (Run12/13's field-of-view change)
            -> (patch - sample_median) / (sample_scale + eps)   [Run13's per-sample robust norm]
            -> model -> sigmoid

    No new threshold, no new epsilon scheme beyond what Run11/12/13 already established --
    sample_median/sample_scale/eps are passed in, never computed here from this single image."""
    kept, confidences = [], []

    for y, x in centers:
        patch = extract_patch(image, y, x, patch_size)
        if patch is None:
            continue

        patch = crop_and_resize_patch(patch)
        patch = (patch.astype(np.float32) - sample_median) / (sample_scale + eps)
        tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        logit = model(tensor).squeeze().item()
        prob = torch.sigmoid(torch.tensor(logit)).item()

        if prob >= 0.5:
            kept.append((int(y), int(x)))
            confidences.append(prob)

    return kept, confidences


def load_model(checkpoint_path, device):
    # Run10 onward all train CellCNNNoClassifierReLU (identical architecture to CellCNN
    # except the ReLU after the first classifier Linear is a genuinely separate module
    # absent from the Sequential, not just skipped in forward() -- this shifts the final
    # Linear layer's state_dict key from classifier.4 to classifier.3). Checkpoints from
    # that point on are only loadable into this class.
    model = CellCNNNoClassifierReLU().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()
    return model


def extract_patch(image, y, x, patch_size=None):
    patch_size = patch_size or config.PATCH_SIZE
    half = patch_size // 2

    y1, y2 = y - half, y + half
    x1, x2 = x - half, x + half

    if y1 < 0 or x1 < 0 or y2 >= image.shape[0] or x2 >= image.shape[1]:
        return None

    return image[y1:y2, x1:x2]


@torch.no_grad()
def classify_centers(model, image, centers, mean, std, device, patch_size=None):
    """Returns (kept_centers, confidences) for centers whose patch classifies
    as a real cell above 0.5 probability."""
    kept, confidences = [], []

    for y, x in centers:
        patch = extract_patch(image, y, x, patch_size)
        if patch is None:
            continue

        patch = (patch.astype(np.float32) - mean) / (std + 1e-8)
        tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        logit = model(tensor).squeeze().item()
        prob = torch.sigmoid(torch.tensor(logit)).item()

        if prob >= 0.5:
            kept.append((int(y), int(x)))
            confidences.append(prob)

    return kept, confidences


def main():
    parser = argparse.ArgumentParser(
        description="Run detector + CNN classifier on a sample using Run13 sample-robust inference."
    )
    parser.add_argument("--sample", required=True, help="Sample name (without .zarr)")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--frame", type=int, default=0, help="Frame index to write predictions for")
    parser.add_argument("--checkpoint", default=str(config.BEST_MODEL_PATH))
    parser.add_argument("--out", default=str(config.OUTPUT_PATH / "predictions.csv"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    dataset = BioHubDataset(config.DATASET_PATH)
    volume = np.asarray(dataset.load_volume(args.sample, split=args.split))

    if args.frame < 0 or args.frame >= len(volume):
        raise ValueError(
            f"Frame index {args.frame} is out of range for sample "
            f"{args.sample!r} with {len(volume)} frames."
        )

    # Build one max-projection per frame and detect candidates across the whole
    # sample. Run13 normalization is defined from the pooled candidate patches
    # of the complete volume, not from one frame.
    projections = [volume[t].max(axis=0) for t in range(len(volume))]

    detector = CellDetector(
        sigma=config.GAUSSIAN_SIGMA,
        threshold_abs=config.DETECTION_THRESHOLD,
        min_distance=config.CELL_RADIUS,
    )

    raw_detections = []
    for t, projection in enumerate(projections):
        centers_t = detector.detect(projection)
        raw_detections.append(centers_t)
        logger.info(
            "Frame %d: detector proposed %d candidate centers",
            t,
            len(centers_t),
        )

    total_candidates = sum(len(c) for c in raw_detections)

    if total_candidates == 0:
        kept, confidences = [], []
        logger.info("No candidate centers found in the sample; skipping CNN inference.")
    else:
        # Compute sample-wide robust normalization exactly once from all
        # crop-resized candidate patches in the volume.
        sample_patches = collect_sample_patches(projections, raw_detections)
        sample_median, sample_scale = compute_sample_robust_stats(sample_patches)

        logger.info(
            "Run13 sample normalization: median=%.6f scale=%.6f "
            "from %d candidate patches",
            sample_median,
            sample_scale,
            len(sample_patches),
        )

        model = load_model(args.checkpoint, device)

        kept, confidences = classify_centers_robust(
            model,
            projections[args.frame],
            raw_detections[args.frame],
            sample_median,
            sample_scale,
            device,
        )

    logger.info(
        "Frame %d: CNN accepted %d / %d candidates",
        args.frame,
        len(kept),
        len(raw_detections[args.frame]),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y", "x", "confidence"])
        for (y, x), conf in zip(kept, confidences):
            writer.writerow([y, x, f"{conf:.4f}"])

    logger.info("Wrote %d predictions to %s", len(kept), out_path)


if __name__ == "__main__":
    main()
