"""
Standalone inference script.

Usage:
    python -m src.predict --sample <sample_name> --split test --out outputs/predictions.csv
    python -m src.predict --sample <sample_name> --frame 40 --checkpoint models/best_model.pth

Loads a trained CellCNN checkpoint, runs the (unchanged, already-working)
CellDetector to propose candidate centers on a max-projection, extracts a
patch around each candidate, classifies each patch, and writes accepted
centers with their confidence to a CSV. This does not modify src/detector.py
or src/tracker.py — it only adds a CNN confidence pass in front of the
existing detector output.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from src import config
from src.dataset import BioHubDataset
from src.detector import CellDetector
from src.logging_utils import setup_logging
from src.model import CellCNN

logger = setup_logging()


def load_model(checkpoint_path, device):
    model = CellCNN().to(device)
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
    parser = argparse.ArgumentParser(description="Run detector + CNN classifier on a sample.")
    parser.add_argument("--sample", required=True, help="Sample name (without .zarr)")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--frame", type=int, default=0, help="Frame index to run on")
    parser.add_argument("--checkpoint", default=str(config.BEST_MODEL_PATH))
    parser.add_argument("--out", default=str(config.OUTPUT_PATH / "predictions.csv"))
    parser.add_argument(
        "--norm-stats",
        default=str(config.BEST_NORM_STATS_PATH),
        help="Path to a small .npz with 'mean'/'std' saved during training. "
        "Defaults to models/norm_stats.npz, which Experiment.promote_to_default() "
        "copies there alongside the promoted checkpoint (see src/experiment.py) — "
        "so by default this always matches whatever checkpoint --checkpoint "
        "points at. If that file doesn't exist, falls back to the image's own "
        "mean/std, which is less accurate than the train-set statistics the "
        "model was actually trained on.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    dataset = BioHubDataset(config.DATASET_PATH)
    volume = np.asarray(dataset.load_volume(args.sample, split=args.split))
    frame = volume[args.frame]
    projection = frame.max(axis=0)
    logger.info("Loaded sample=%s frame=%d, projection shape=%s", args.sample, args.frame, projection.shape)

    norm_stats_path = Path(args.norm_stats)
    if norm_stats_path.exists():
        stats = np.load(norm_stats_path)
        mean, std = float(stats["mean"]), float(stats["std"])
    else:
        mean, std = float(projection.mean()), float(projection.std())
        logger.warning(
            "%s not found; using this image's own mean/std. "
            "For best accuracy, run training first so norm_stats.npz gets "
            "promoted, or pass --norm-stats explicitly.",
            norm_stats_path,
        )

    detector = CellDetector(
        sigma=config.GAUSSIAN_SIGMA,
        threshold_abs=config.DETECTION_THRESHOLD,
        min_distance=config.CELL_RADIUS,
    )
    centers = detector.detect(projection)
    logger.info("Detector proposed %d candidate centers", len(centers))

    model = load_model(args.checkpoint, device)
    kept, confidences = classify_centers(model, projection, centers, mean, std, device)
    logger.info("CNN accepted %d / %d candidates", len(kept), len(centers))

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
