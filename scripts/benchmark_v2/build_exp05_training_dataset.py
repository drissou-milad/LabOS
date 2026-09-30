"""
Experiment #5, stage 1: GT-aware training-data construction.

Builds a labeled candidate dataset (positive / negative / ambiguous, per the decision recorded
in this project's Experiment #5 audit) for a LATER CellCNN training stage. This script does
NOT train anything, does NOT import torch, does NOT import CellCNN, and does NOT create a
checkpoint -- it only produces manifest CSVs, split assignments, and extracted 32x32 patch
files on disk, so the actual training step (a separate, future piece of work) has a fully
audited, GT-derived dataset to consume.

LABELING POLICY (frozen per the Experiment #5 audit decision -- do not change these without a
new, explicit decision record):
    distance <= --positive-radius (default 5px)  -> positive
    distance >  --negative-radius (default 10px) -> negative
    --positive-radius < distance <= --negative-radius -> ambiguous (excluded from training)

DISTANCE METHOD: for every candidate, independently, nearest-GT distance is computed via
scipy.spatial.distance.cdist + .min(axis=1) against ONLY the GT points in that candidate's own
frame (same sample, same timepoint). This is NOT Hungarian/optimal-assignment matching --
matching_v2.py is intentionally not imported or used here; a 1:1 assignment would "spend"
candidates on a globally optimal pairing, which is the wrong operation for per-candidate
labeling (see the audit script's docstring for the same reasoning).

REUSED, UNMODIFIED APIS:
    - src.dataset.BioHubDataset          (data loading only)
    - scripts/experiments/exp01_adaptive_threshold/adaptive_detector.AdaptiveCellDetector
    - src.predict.extract_patch          (patch extraction; always called with an explicit
                                           patch_size argument so it never falls back to
                                           config.PATCH_SIZE)
    - the confirmed nodes/props/<name>/values GT loader (duplicated here, same isolation
      rationale as every prior Experiment #3/#4/audit script)

NOT IMPORTED, ON PURPOSE: torch, src.model.CellCNN, src.train, matching_v2. This script has no
mechanism by which it could train a model even by accident -- there is no training loop, no
optimizer, no loss function, and no model instantiation anywhere in this file.

Does NOT modify: src/dataset.py, src/train.py, src/model.py, src/predict.py, src/config.py,
matching_v2.py, run_benchmark_v2.py, run_benchmark_v2_adaptive.py,
run_benchmark_v2_min_distance_sweep.py, run_benchmark_v2_otsu_min_distance.py,
adaptive_detector.py, or any frozen Experiment #1-#4 / audit result.

Usage:
    export BIOHUB_DATASET_PATH=/path/to/biohub-cell-tracking-during-development

    python scripts/benchmark_v2/build_exp05_training_dataset.py \\
        --out-dir results/exp05_training_dataset/run01_smoke \\
        --limit-samples 3 \\
        --train-fraction 0.70 --val-fraction 0.15 --test-fraction 0.15 --seed 42

Output (under --out-dir only):
    manifest.csv          one row per label-eligible candidate (see column list below)
    split_manifest.csv    sample_id, split
    dataset_summary.txt   full provenance + all requested counts
    patches/<split>/<positive|negative>/*.npy
    inspection/<positive|negative|ambiguous>/*.npy  (small deterministic QC subset)
"""

import argparse
import csv
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.spatial.distance import cdist

from src.dataset import BioHubDataset          # unmodified; data loading only
from src.predict import extract_patch          # unmodified; pure function, patch_size always explicit

ADAPTIVE_DETECTOR_DIR = (
    PROJECT_ROOT / "scripts" / "experiments" / "exp01_adaptive_threshold"
)
sys.path.insert(0, str(ADAPTIVE_DETECTOR_DIR))
from adaptive_detector import AdaptiveCellDetector  # unmodified, reused as-is

REQUIRED_OUTPUT_PARENT = "exp05_training_dataset"

import os
DEFAULT_DATASET_PATH = Path(os.environ.get(
    "BIOHUB_DATASET_PATH",
    r"D:\Datasets\BioHub\biohub-cell-tracking-during-development",
))

MANIFEST_COLUMNS = [
    "sample_id", "frame_idx", "candidate_y", "candidate_x",
    "nearest_gt_y", "nearest_gt_x", "nearest_gt_distance_px",
    "label", "split", "patch_status", "patch_path",
    "sigma", "threshold_method", "threshold_floor", "min_distance", "d_max_px",
    "patch_size", "detector_class", "adaptive_detector_sha256", "run_timestamp_utc",
]

PROTECTED_LABEL_KEYWORDS = None  # placeholder, not used -- keeping module free of any training hooks


# --------------------------------------------------------------------------
# Pure, independently-testable helper functions
# --------------------------------------------------------------------------

def load_ground_truth_points(dataset, sample):
    """Confirmed schema (nodes/props/<name>/values), verified on sample 44b6_0113de3b.
    Duplicated here, same isolation rationale as every prior Experiment script."""
    graph = dataset.load_graph(sample)
    props = graph["nodes"]["props"]
    ts = np.asarray(props["t"]["values"][:])
    ys = np.asarray(props["y"]["values"][:])
    xs = np.asarray(props["x"]["values"][:])

    n_ids = graph["nodes"]["ids"].shape[0]
    if not (len(ts) == len(ys) == len(xs) == n_ids):
        raise ValueError(
            f"Sample {sample}: nodes/props array lengths (t={len(ts)}, y={len(ys)}, "
            f"x={len(xs)}) don't match nodes/ids length ({n_ids}) -- aborting rather than "
            f"labeling against untrustworthy ground truth."
        )
    by_frame = {}
    for t, y, x in zip(ts, ys, xs):
        by_frame.setdefault(int(t), []).append((float(y), float(x)))
    return {t: np.array(pts) for t, pts in by_frame.items()}


def sha256_of_file(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception as e:
        return f"<could not hash: {e}>"


def nearest_gt_for_candidates(candidates, gt_pts):
    """candidates: (N,2) array of (y,x). gt_pts: (M,2) array of (y,x), M>=1.
    Returns (nearest_dist (N,), nearest_gt_idx (N,)). Frame-local by construction: caller must
    only pass GT points from the SAME frame as candidates."""
    D = cdist(np.asarray(candidates), gt_pts)  # (N, M)
    nearest_idx = D.argmin(axis=1)
    nearest_dist = D.min(axis=1)
    return nearest_dist, nearest_idx


def label_for_distance(distance, positive_radius, negative_radius):
    """Pure labeling function -- the single source of truth for the frozen labeling policy.
    distance <= positive_radius -> 'positive'
    distance >  negative_radius -> 'negative'
    otherwise (positive_radius < distance <= negative_radius) -> 'ambiguous'"""
    if distance <= positive_radius:
        return "positive"
    if distance > negative_radius:
        return "negative"
    return "ambiguous"


def split_samples(sample_ids, train_fraction, val_fraction, test_fraction, seed):
    """Deterministic SAMPLE-level split (never patch-level). Returns {sample_id: split_name}.
    Same (sample_ids, fractions, seed) always produces the same assignment -- uses a seeded
    RNG shuffle, not Python's unseeded hash-based ordering, and sorts the input first so the
    starting order itself is deterministic regardless of upstream iteration order."""
    ids = sorted(sample_ids)
    rng = np.random.default_rng(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(train_fraction * n)
    n_val = round(val_fraction * n)
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    n_test = n - n_train - n_val  # remainder, never negative given the clamps above

    assignment = {}
    for i, sid in enumerate(shuffled):
        if i < n_train:
            assignment[sid] = "train"
        elif i < n_train + n_val:
            assignment[sid] = "val"
        else:
            assignment[sid] = "test"
    return assignment


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def build_dataset(args, dataset, samples, out_dir, run_timestamp_utc, detector_sha256):
    split_assignment = split_samples(
        samples, args.train_fraction, args.val_fraction, args.test_fraction, args.seed
    )

    manifest_rows = []
    counts = {
        "total_frames": 0, "total_gt": 0, "total_candidates": 0,
        "zero_gt_candidates_excluded": 0,
        "positive": 0, "negative": 0, "ambiguous": 0,
        "positive_edge_excluded": 0, "negative_edge_excluded": 0, "ambiguous_edge_excluded": 0,
        "positive_saved": 0, "negative_saved": 0,
    }

    for sample in samples:
        gt_by_frame = load_ground_truth_points(dataset, sample)
        volume = np.asarray(dataset.load_volume(sample))
        split = split_assignment[sample]

        detector = AdaptiveCellDetector(
            sigma=args.sigma,
            min_distance=args.min_distance,
            threshold_method=args.threshold_method,
            threshold_floor=args.threshold_floor,
        )
        raw_detections = detector.detect_volume(volume)

        for t in range(volume.shape[0]):
            counts["total_frames"] += 1
            gt_pts = gt_by_frame.get(t, np.empty((0, 2)))
            pred_pts = raw_detections[t]
            counts["total_gt"] += len(gt_pts)
            counts["total_candidates"] += len(pred_pts)

            if len(pred_pts) == 0:
                continue

            if len(gt_pts) == 0:
                counts["zero_gt_candidates_excluded"] += len(pred_pts)
                continue

            mip = np.asarray(volume[t]).max(axis=0)
            nearest_dist, nearest_idx = nearest_gt_for_candidates(pred_pts, gt_pts)

            for (y, x), dist, gi in zip(pred_pts, nearest_dist, nearest_idx):
                label = label_for_distance(dist, args.positive_radius, args.negative_radius)
                gy, gx = gt_pts[gi]
                counts[label] += 1

                patch_status = "not_saved"
                patch_path = ""

                patch = extract_patch(mip, int(y), int(x), patch_size=args.patch_size)
                if patch is None:
                    patch_status = "edge_excluded"
                    counts[f"{label}_edge_excluded"] += 1
                elif label in ("positive", "negative"):
                    rel_dir = Path("patches") / split / label
                    (out_dir / rel_dir).mkdir(parents=True, exist_ok=True)
                    fname = f"{sample}_{t}_{int(y)}_{int(x)}.npy"
                    np.save(out_dir / rel_dir / fname, patch.astype(np.float32))
                    patch_status = "train_saved"
                    patch_path = str(rel_dir / fname)
                    counts[f"{label}_saved"] += 1
                # label == "ambiguous" and patch is not None: leave as "not_saved" here;
                # inspection-set selection (below, after the main loop) decides which
                # ambiguous candidates actually get a patch written to inspection/ambiguous/.

                manifest_rows.append({
                    "sample_id": sample, "frame_idx": t,
                    "candidate_y": int(y), "candidate_x": int(x),
                    "nearest_gt_y": float(gy), "nearest_gt_x": float(gx),
                    "nearest_gt_distance_px": float(dist),
                    "label": label, "split": split,
                    "patch_status": patch_status, "patch_path": patch_path,
                    "sigma": args.sigma, "threshold_method": args.threshold_method,
                    "threshold_floor": args.threshold_floor, "min_distance": args.min_distance,
                    "d_max_px": args.d_max, "patch_size": args.patch_size,
                    "detector_class": "AdaptiveCellDetector",
                    "adaptive_detector_sha256": detector_sha256,
                    "run_timestamp_utc": run_timestamp_utc,
                    # kept only for inspection-set selection below, stripped before CSV write:
                    "_mip_ref": mip, "_patch_obj": patch,
                })

    return manifest_rows, split_assignment, counts


def write_inspection_set(manifest_rows, out_dir, max_pos, max_neg, max_amb):
    """Deterministic selection: stable-sorted by (sample_id, frame_idx, candidate_y,
    candidate_x), first N per category with a successfully-extracted patch. Does not respect
    train/val/test split boundaries -- this is a visual QC set, not training data."""
    def sort_key(r):
        return (r["sample_id"], r["frame_idx"], r["candidate_y"], r["candidate_x"])

    eligible = [r for r in manifest_rows if r["_patch_obj"] is not None]
    eligible.sort(key=sort_key)

    written = {"positive": 0, "negative": 0, "ambiguous": 0}
    caps = {"positive": max_pos, "negative": max_neg, "ambiguous": max_amb}

    for r in eligible:
        label = r["label"]
        if written[label] >= caps[label]:
            continue
        rel_dir = Path("inspection") / label
        (out_dir / rel_dir).mkdir(parents=True, exist_ok=True)
        fname = f"{r['sample_id']}_{r['frame_idx']}_{r['candidate_y']}_{r['candidate_x']}.npy"
        np.save(out_dir / rel_dir / fname, r["_patch_obj"].astype(np.float32))
        written[label] += 1

    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--threshold-method", choices=["otsu"], default="otsu")
    parser.add_argument("--threshold-floor", type=float, default=10.0)
    parser.add_argument("--min-distance", type=int, default=5)
    parser.add_argument("--d-max", type=float, default=15.0,
                         help="Recorded for provenance only; NOT used for labeling here")
    parser.add_argument("--positive-radius", type=float, default=5.0)
    parser.add_argument("--negative-radius", type=float, default=10.0)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-inspection-positive", type=int, default=50)
    parser.add_argument("--max-inspection-negative", type=int, default=50)
    parser.add_argument("--max-inspection-ambiguous", type=int, default=25)
    parser.add_argument("--allow-nonempty-out-dir", action="store_true",
                         help="Permit writing into a non-empty --out-dir. Even with this flag, "
                              "an existing manifest.csv/dataset_summary.txt/split_manifest.csv "
                              "inside it will still refuse to be overwritten.")
    args = parser.parse_args()

    if not (args.positive_radius < args.negative_radius):
        raise SystemExit(
            f"--positive-radius ({args.positive_radius}) must be strictly less than "
            f"--negative-radius ({args.negative_radius})."
        )
    frac_sum = args.train_fraction + args.val_fraction + args.test_fraction
    if abs(frac_sum - 1.0) > 1e-6:
        raise SystemExit(
            f"--train-fraction + --val-fraction + --test-fraction must sum to 1.0, got {frac_sum}"
        )

    out_dir = Path(args.out_dir)
    if out_dir.parent.name != REQUIRED_OUTPUT_PARENT:
        raise SystemExit(
            f"Refusing to write outside results/{REQUIRED_OUTPUT_PARENT}/<run_name>/ -- got "
            f"'{out_dir}'."
        )
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_nonempty_out_dir:
        raise SystemExit(
            f"Refusing to write into non-empty existing directory {out_dir} "
            f"(pass --allow-nonempty-out-dir to override)."
        )
    for protected in ["manifest.csv", "dataset_summary.txt", "split_manifest.csv"]:
        if (out_dir / protected).exists():
            raise SystemExit(
                f"Refusing to overwrite existing {out_dir / protected} under any circumstance "
                f"-- choose a new --out-dir for a new run."
            )
    out_dir.mkdir(parents=True, exist_ok=True)

    detector_sha256 = sha256_of_file(ADAPTIVE_DETECTOR_DIR / "adaptive_detector.py")
    run_timestamp_utc = datetime.now(timezone.utc).isoformat()

    dataset = BioHubDataset(DEFAULT_DATASET_PATH)
    all_samples = dataset.train_samples
    samples = all_samples[: args.limit_samples] if args.limit_samples else all_samples

    print("Experiment #5 stage 1: GT-aware training-data construction")
    print(f"sigma={args.sigma}, threshold_method={args.threshold_method}, "
          f"threshold_floor={args.threshold_floor}, min_distance={args.min_distance}, "
          f"d_max={args.d_max} (recorded only)")
    print(f"positive_radius<={args.positive_radius}px, negative_radius>{args.negative_radius}px, "
          f"ambiguous in between")
    print(f"Samples ({len(samples)}): {samples}")
    print("(No training, no checkpoint, no model import anywhere in this script.)\n")

    manifest_rows, split_assignment, counts = build_dataset(
        args, dataset, samples, out_dir, run_timestamp_utc, detector_sha256
    )

    inspection_written = write_inspection_set(
        manifest_rows, out_dir,
        args.max_inspection_positive, args.max_inspection_negative, args.max_inspection_ambiguous,
    )

    # Strip internal-only fields before writing manifest.csv
    clean_rows = [{k: v for k, v in r.items() if k in MANIFEST_COLUMNS} for r in manifest_rows]
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(clean_rows)

    with open(out_dir / "split_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "split"])
        for sid in sorted(split_assignment):
            w.writerow([sid, split_assignment[sid]])

    n_pos_lost_edge = counts["positive_edge_excluded"]
    n_neg_lost_edge = counts["negative_edge_excluded"]
    final_pos = counts["positive_saved"]
    final_neg = counts["negative_saved"]
    ratio = (final_neg / final_pos) if final_pos else float("inf")

    train_ids = sorted(s for s, sp in split_assignment.items() if sp == "train")
    val_ids = sorted(s for s, sp in split_assignment.items() if sp == "val")
    test_ids = sorted(s for s, sp in split_assignment.items() if sp == "test")

    summary_lines = [
        "Experiment #5 stage 1: GT-aware training-data construction",
        "=" * 78,
        "PROVENANCE",
        f"  run_timestamp_utc = {run_timestamp_utc}",
        f"  command_line = {' '.join(sys.argv)}",
        f"  adaptive_detector.py SHA-256 = {detector_sha256}",
        f"  dataset_path = {DEFAULT_DATASET_PATH}",
        f"  samples (n={len(samples)}) = {samples}",
        f"  split_seed = {args.seed}",
        "  EXPERIMENTAL PARAMETERS (explicit CLI literals, never read from config.py):",
        f"    sigma={args.sigma}, threshold_method={args.threshold_method}, "
        f"threshold_floor={args.threshold_floor}, min_distance={args.min_distance}, "
        f"d_max={args.d_max} (recorded only, not used for labeling)",
        f"    positive_radius<={args.positive_radius}px, negative_radius>{args.negative_radius}px",
        f"    patch_size={args.patch_size}",
        "=" * 78,
        "",
        "--- Sample split ---",
        f"  train ({len(train_ids)}): {train_ids}",
        f"  val   ({len(val_ids)}): {val_ids}",
        f"  test  ({len(test_ids)}): {test_ids}",
        "",
        "--- Counts ---",
        f"  Frames processed = {counts['total_frames']}",
        f"  Total GT points = {counts['total_gt']}",
        f"  Total detector candidates = {counts['total_candidates']}",
        f"  Candidates excluded (zero GT in frame) = {counts['zero_gt_candidates_excluded']}",
        f"  Positive-labeled candidates = {counts['positive']}",
        f"  Negative-labeled candidates = {counts['negative']}",
        f"  Ambiguous-labeled candidates (excluded from training) = {counts['ambiguous']}",
        f"  Positives lost to patch boundary (edge_excluded) = {n_pos_lost_edge}",
        f"  Negatives lost to patch boundary (edge_excluded) = {n_neg_lost_edge}",
        f"  Ambiguous lost to patch boundary (edge_excluded, informational only) = "
        f"{counts['ambiguous_edge_excluded']}",
        f"  FINAL usable positive training patches saved = {final_pos}",
        f"  FINAL usable negative training patches saved = {final_neg}",
        f"  Final negative:positive ratio = {ratio:.2f}:1" if final_pos else
        "  Final negative:positive ratio = undefined (zero positives saved)",
        "",
        "--- Inspection set written (deterministic, does not respect split boundaries) ---",
        f"  positive: {inspection_written['positive']} / cap {args.max_inspection_positive}",
        f"  negative: {inspection_written['negative']} / cap {args.max_inspection_negative}",
        f"  ambiguous: {inspection_written['ambiguous']} / cap {args.max_inspection_ambiguous}",
        "",
        "TRAINING WAS NOT PERFORMED. NO CHECKPOINT WAS CREATED. This script imports no model, "
        "no torch training utilities, and contains no training loop.",
    ]
    summary_text = "\n".join(summary_lines)
    (out_dir / "dataset_summary.txt").write_text(summary_text)
    print("\n" + summary_text)


if __name__ == "__main__":
    main()
