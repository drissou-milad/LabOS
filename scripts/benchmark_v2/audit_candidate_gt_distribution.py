"""
Experiment #5 prerequisite: candidate-vs-GT distance distribution audit.

READ-ONLY. Does not train anything, does not create a checkpoint, does not modify
src/dataset.py, src/train.py, src/model.py, src/predict.py, matching_v2.py, or any existing
benchmark script or frozen result. Imports AdaptiveCellDetector and BioHubDataset UNMODIFIED,
purely to generate candidates and load GT for analysis.

WHAT THIS COMPUTES, AND WHY IT'S DIFFERENT FROM matching_v2.gated_match: Benchmark v2's gated
matching answers "under a 1:1 optimal assignment, how many candidates count as TP at d_max?".
This audit answers a different, more basic question that a 1:1 assignment can obscure: for
EVERY candidate independently, how far is it from its single nearest GT point (many-to-one,
no assignment competition)? This is the right question for a training-label design decision,
because a future GT-based patch labeler needs to decide "positive" or "negative" per candidate,
not per matched-pair -- it can't rely on Hungarian assignment "spending" a candidate on the
globally optimal solution the way benchmark scoring does.

Per-frame only: a candidate's nearest-GT distance is computed only against GT points in its
OWN frame (same sample, same timepoint) -- never across frames.

Uses the confirmed frozen Otsu reference configuration by default (sigma=2, threshold_floor=10,
min_distance=5, threshold_method=otsu) -- CLI-overridable literals, no src.config reads for
detection parameters, consistent with every script since the provenance investigation earlier
in this project.

Does NOT pick a training-labeling radius. Reports distributions at multiple candidate radii
(1/2/3/5/7/10/15px) and multiple GT-clustering radii (3/5/7/10/15px) so that decision can be
made from evidence afterward, not embedded silently in this script.

Usage:
    export BIOHUB_DATASET_PATH=/path/to/biohub-cell-tracking-during-development

    # Quick subset (matches prior smoke-test convention)
    python scripts/benchmark_v2/audit_candidate_gt_distribution.py \\
        --out-dir results/exp05_audit/audit_first3 --limit-samples 3

    # Larger subset, recommended before finalizing any labeling-radius decision (see note in
    # this script's final printed report about why 3 samples may be too few for GT-side stats)
    python scripts/benchmark_v2/audit_candidate_gt_distribution.py \\
        --out-dir results/exp05_audit/audit_first20 --limit-samples 20

Outputs (under the given --out-dir only):
    candidate_distances.csv   one row per candidate: sample, frame, y, x, nearest_gt_distance_px
    gt_neighbor_counts.csv    one row per GT point: sample, frame, y, x, and candidate counts
                               within each of {3,5,7,10,15}px
    audit_summary.txt         everything requested in items 1-8 of the audit spec, plus
                               full provenance
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

from src.dataset import BioHubDataset  # unmodified; used only to load data

ADAPTIVE_DETECTOR_DIR = (
    PROJECT_ROOT / "scripts" / "experiments" / "exp01_adaptive_threshold"
)
sys.path.insert(0, str(ADAPTIVE_DETECTOR_DIR))
from adaptive_detector import AdaptiveCellDetector  # unmodified, reused as-is

REQUIRED_OUTPUT_PARENT = "exp05_audit"

import os
DEFAULT_DATASET_PATH = Path(os.environ.get(
    "BIOHUB_DATASET_PATH",
    r"D:\Datasets\BioHub\biohub-cell-tracking-during-development",
))

CANDIDATE_RADII = [1, 2, 3, 5, 7, 10, 15]  # cumulative "<= this many px" bins, plus a ">15" tail
POSITIVE_RADII_TO_REPORT = [3, 5, 7, 10]
NEGATIVE_THRESHOLDS_TO_REPORT = [10, 15]
GT_CLUSTER_RADII = [3, 5, 7, 10, 15]


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
            f"auditing against untrustworthy ground truth."
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


def percentile_block(arr):
    if len(arr) == 0:
        return {k: float("nan") for k in ["min", "median", "mean", "p25", "p75", "p90", "p95", "p99", "max"]}
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit-samples", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--threshold-floor", type=float, default=10.0)
    parser.add_argument("--min-distance", type=int, default=5)
    parser.add_argument("--threshold-method", choices=["otsu"], default="otsu",
                         help="Restricted to the confirmed reference method for this audit")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.parent.name != REQUIRED_OUTPUT_PARENT:
        raise SystemExit(
            f"Refusing to write outside results/{REQUIRED_OUTPUT_PARENT}/<name>/ -- got "
            f"'{out_dir}'. Keep audit outputs isolated from benchmark/experiment results."
        )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"Refusing to write into non-empty existing directory {out_dir}.")
    out_dir.mkdir(parents=True, exist_ok=True)

    detector_sha256 = sha256_of_file(ADAPTIVE_DETECTOR_DIR / "adaptive_detector.py")
    run_timestamp_utc = datetime.now(timezone.utc).isoformat()

    dataset = BioHubDataset(DEFAULT_DATASET_PATH)
    all_samples = dataset.train_samples
    samples = all_samples[: args.limit_samples] if args.limit_samples else all_samples

    print("Experiment #5 prerequisite audit: candidate-vs-GT distance distribution")
    print(f"sigma={args.sigma}, threshold_method={args.threshold_method}, "
          f"threshold_floor={args.threshold_floor}, min_distance={args.min_distance}")
    print(f"Samples inspected ({len(samples)}): {samples}")
    print("(Read-only: no training, no checkpoint, no modification to any existing file.)\n")

    candidate_rows = []       # sample, frame, y, x, nearest_gt_distance_px
    gt_neighbor_rows = []     # sample, frame, y, x, count_within_3/5/7/10/15
    total_frames = 0

    for si, sample in enumerate(samples):
        print(f"[{si + 1}/{len(samples)}] {sample}", flush=True)
        gt_by_frame = load_ground_truth_points(dataset, sample)
        volume = np.asarray(dataset.load_volume(sample))

        detector = AdaptiveCellDetector(
            sigma=args.sigma,
            min_distance=args.min_distance,
            threshold_method=args.threshold_method,
            threshold_floor=args.threshold_floor,
        )
        raw_detections = detector.detect_volume(volume)

        for t in range(volume.shape[0]):
            total_frames += 1
            gt_pts = gt_by_frame.get(t, np.empty((0, 2)))
            pred_pts = raw_detections[t]

            if len(pred_pts) > 0 and len(gt_pts) > 0:
                D = cdist(np.asarray(pred_pts), gt_pts)  # (n_pred, n_gt)
                nearest_dist = D.min(axis=1)
                for (y, x), dist in zip(pred_pts, nearest_dist):
                    candidate_rows.append({
                        "sample": sample, "frame": t, "y": int(y), "x": int(x),
                        "nearest_gt_distance_px": float(dist),
                    })
                for gi, (gy, gx) in enumerate(gt_pts):
                    row = {"sample": sample, "frame": t, "y": gy, "x": gx}
                    for r in GT_CLUSTER_RADII:
                        row[f"n_candidates_within_{r}px"] = int((D[:, gi] <= r).sum())
                    gt_neighbor_rows.append(row)
            elif len(pred_pts) > 0 and len(gt_pts) == 0:
                # Candidates exist but no GT in this frame -- record as "no GT to compare
                # against", excluded from distance stats, but the existence is worth counting.
                for (y, x) in pred_pts:
                    candidate_rows.append({
                        "sample": sample, "frame": t, "y": int(y), "x": int(x),
                        "nearest_gt_distance_px": float("nan"),
                    })
            elif len(pred_pts) == 0 and len(gt_pts) > 0:
                for gi, (gy, gx) in enumerate(gt_pts):
                    row = {"sample": sample, "frame": t, "y": gy, "x": gx}
                    for r in GT_CLUSTER_RADII:
                        row[f"n_candidates_within_{r}px"] = 0
                    gt_neighbor_rows.append(row)
            # both empty: nothing to record

    # ---- write raw per-candidate / per-GT CSVs ----
    if candidate_rows:
        with open(out_dir / "candidate_distances.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(candidate_rows[0].keys()))
            w.writeheader()
            w.writerows(candidate_rows)
    if gt_neighbor_rows:
        with open(out_dir / "gt_neighbor_counts.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(gt_neighbor_rows[0].keys()))
            w.writeheader()
            w.writerows(gt_neighbor_rows)

    # ---- compute summary stats ----
    all_dist = np.array([r["nearest_gt_distance_px"] for r in candidate_rows
                          if not np.isnan(r["nearest_gt_distance_px"])])
    n_candidates_no_gt_in_frame = sum(
        1 for r in candidate_rows if np.isnan(r["nearest_gt_distance_px"])
    )
    total_candidates = len(candidate_rows)
    total_gt = len(gt_neighbor_rows)

    # item 3: bin counts
    bin_lines = []
    prev_cum = 0
    for r in CANDIDATE_RADII:
        cum = int((all_dist <= r).sum())
        bin_lines.append(f"  <= {r:>2d} px : {cum - prev_cum:8d}  (cumulative <= {r}px: {cum}, "
                          f"{100*cum/len(all_dist):.3f}% of distance-eligible candidates)")
        prev_cum = cum
    n_over_15 = int((all_dist > 15).sum())
    bin_lines.append(f"  >  15 px : {n_over_15:8d}  ({100*n_over_15/len(all_dist):.3f}%)")

    stats = percentile_block(all_dist)

    # item 5/6/8: positive/negative/ambiguous matrix
    matrix_lines = []
    for pos_r in POSITIVE_RADII_TO_REPORT:
        for neg_t in NEGATIVE_THRESHOLDS_TO_REPORT:
            if neg_t <= pos_r:
                continue
            n_pos = int((all_dist <= pos_r).sum())
            n_neg = int((all_dist > neg_t).sum())
            n_amb = len(all_dist) - n_pos - n_neg
            pct_pos = 100 * n_pos / len(all_dist)
            pct_neg = 100 * n_neg / len(all_dist)
            pct_amb = 100 * n_amb / len(all_dist)
            ratio = f"{n_pos}:{n_neg}" if n_neg > 0 else f"{n_pos}:0 (undefined)"
            imbalance = (n_neg / n_pos) if n_pos > 0 else float("inf")
            matrix_lines.append(
                f"  positive<= {pos_r:>2d}px, negative> {neg_t:>2d}px : "
                f"pos={n_pos:7d} ({pct_pos:6.3f}%)  neg={n_neg:7d} ({pct_neg:6.3f}%)  "
                f"ambiguous={n_amb:7d} ({pct_amb:6.3f}%)  "
                f"neg:pos ratio={imbalance:.1f}:1"
            )

    # GT clustering (item on multiple candidates near same GT)
    gt_cluster_lines = []
    for r in GT_CLUSTER_RADII:
        col = f"n_candidates_within_{r}px"
        counts = [row[col] for row in gt_neighbor_rows]
        n0 = sum(1 for c in counts if c == 0)
        n1 = sum(1 for c in counts if c == 1)
        n2p = sum(1 for c in counts if c >= 2)
        gt_cluster_lines.append(
            f"  within {r:>2d}px: 0 candidates={n0:5d} ({100*n0/total_gt:.2f}%)   "
            f"1 candidate={n1:5d} ({100*n1/total_gt:.2f}%)   "
            f"2+ candidates={n2p:5d} ({100*n2p/total_gt:.2f}%)"
        )

    summary_lines = [
        "Experiment #5 prerequisite audit: candidate-vs-GT distance distribution",
        "=" * 78,
        "PROVENANCE",
        f"  run_timestamp_utc = {run_timestamp_utc}",
        f"  command_line = {' '.join(sys.argv)}",
        f"  adaptive_detector.py SHA-256 = {detector_sha256}",
        f"  parameters: sigma={args.sigma}, threshold_method={args.threshold_method}, "
        f"threshold_floor={args.threshold_floor}, min_distance={args.min_distance}",
        f"  dataset_path = {DEFAULT_DATASET_PATH}",
        f"  samples inspected (n={len(samples)}): {samples}",
        f"  total frames inspected: {total_frames}",
        "=" * 78,
        "",
        "--- Item 1-2: totals ---",
        f"  Total GT points = {total_gt}",
        f"  Total detector candidates = {total_candidates}",
        f"  Candidates in frames with zero GT points (excluded from distance stats) = "
        f"{n_candidates_no_gt_in_frame}",
        f"  Candidates with a computable nearest-GT distance = {len(all_dist)}",
        "",
        "--- Item 3: candidate counts by distance-to-nearest-GT bin ---",
        *bin_lines,
        "",
        "--- Item 4: candidate distance distribution statistics (px) ---",
        f"  min={stats['min']:.3f}  median={stats['median']:.3f}  mean={stats['mean']:.3f}  "
        f"p25={stats['p25']:.3f}  p75={stats['p75']:.3f}  p90={stats['p90']:.3f}  "
        f"p95={stats['p95']:.3f}  p99={stats['p99']:.3f}  max={stats['max']:.3f}",
        "",
        "--- Items 5/6/7/8: positive / negative / ambiguous under candidate labeling rules ---",
        "  (pairs where negative threshold <= positive radius are skipped as incoherent)",
        *matrix_lines,
        "",
        "--- GT-side clustering: candidates found near each GT point, at several radii ---",
        *gt_cluster_lines,
        "",
        "NOTE: this script deliberately does not select a training-labeling radius -- that",
        "decision is left to the analysis step, using the distributions above as evidence.",
    ]
    summary_text = "\n".join(summary_lines)
    (out_dir / "audit_summary.txt").write_text(summary_text)
    print("\n" + summary_text)


if __name__ == "__main__":
    main()
