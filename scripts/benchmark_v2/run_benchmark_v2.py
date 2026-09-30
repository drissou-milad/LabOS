"""
Benchmark v2: distance-gated detection evaluation (TP / FP / FN / precision / recall / F1),
d_max = 15px by default.

Entirely separate from scripts/labos_diagnostic_benchmark.py (Benchmark v1). Does not import
from it, does not write to results/benchmark_baseline_v1.csv, does not modify src/detector.py.
Scores the frozen-baseline CellDetector (config.GAUSSIAN_SIGMA / DETECTION_THRESHOLD /
CELL_RADIUS, unmodified) by default -- this is a NEW SCORING METHODOLOGY applied to the SAME
unmodified detector, not a new algorithm. (The --detector flag exists so this same evaluation
methodology can later be pointed at Experiment #1's AdaptiveCellDetector or any future
candidate, without writing a third near-duplicate runner.)

Why this exists, in one sentence: Benchmark v1's uncapped Hungarian matching cannot distinguish
"detector found every real cell, just slightly off" from "detector missed real cells entirely
and got lucky pairing GT points with distant unrelated candidates" -- both currently show up as
some number of large distances baked into a mean/median, with coverage reading 100% either way.
Benchmark v2 makes that distinction explicit via a hard distance gate.

Usage:
    export BIOHUB_DATASET_PATH=/path/to/biohub-cell-tracking-during-development

    # Smoke test
    python scripts/benchmark_v2/run_benchmark_v2.py \\
        --out results/benchmark_v2/benchmark_v2_smoke.csv --limit-samples 3

    # Full run
    python scripts/benchmark_v2/run_benchmark_v2.py \\
        --out results/benchmark_v2/benchmark_v2_baseline_detector.csv

Output columns (one row per sample), MICRO-averaged (TP/FP/FN summed across all frames of the
sample, then precision/recall/F1 computed once from the summed counts -- not averaged
per-frame rates, which would be undefined/misleading on empty frames):
    sample, n_frames, d_max_px, n_gt_total, n_pred_total,
    tp, fp, fn, precision, recall, f1,
    mean_error_px_tp_only, median_error_px_tp_only,
    frac_le_5px_tp_only, frac_le_10px_tp_only, frac_le_15px_tp_only

"frac_le_15px_tp_only" will always be 1.0 when d_max=15, by construction (a TP is defined as
<=d_max) -- it's included anyway so this table has the same shape when d_max is changed via
--d-max, and so <=5px/<=10px/<=15px are reported consistently as requested.

Also writes a per-frame file (same stem + "_frames.csv"), same rationale as Benchmark v1: some
failure modes (one catastrophic frame) are invisible at sample granularity.
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src import config
from src.dataset import BioHubDataset
from src.detector import CellDetector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matching_v2 import gated_match, precision_recall_f1

RESERVED_FILENAMES = {"benchmark_baseline_v1.csv"}  # v2 must never write to v1's frozen file


def load_ground_truth_points(dataset, sample):
    """
    Confirmed against the real dataset via scripts/benchmark_v2/inspect_geff_schema.py on
    sample 44b6_0113de3b: node properties live under nodes/props/<name>/values, not flat
    nodes/<name> (matches the geff spec: https://liveimagetrackingtools.org/geff/v1.1.4.1.1/specification/).
    Axes metadata on that sample confirms "t" is the time axis and "x"/"y"/"z" are spatial.

    NOTE: this fix is applied here only (Benchmark v2). scripts/labos_diagnostic_benchmark.py
    (Benchmark v1) and scripts/experiments/exp01_adaptive_threshold/run_benchmark_adaptive.py
    still use the old flat-schema assumption and were not touched -- they need the same fix
    before being trusted, but that's out of scope for this change.
    """
    graph = dataset.load_graph(sample)
    props = graph["nodes"]["props"]
    ts = np.asarray(props["t"]["values"][:])
    ys = np.asarray(props["y"]["values"][:])
    xs = np.asarray(props["x"]["values"][:])

    n_ids = graph["nodes"]["ids"].shape[0]
    if not (len(ts) == len(ys) == len(xs) == n_ids):
        raise ValueError(
            f"Sample {sample}: nodes/props array lengths (t={len(ts)}, y={len(ys)}, "
            f"x={len(xs)}) don't match nodes/ids length ({n_ids}) -- ground-truth points "
            f"can't be trusted. Re-run scripts/benchmark_v2/inspect_geff_schema.py on this "
            f"sample and check property names/shapes by hand before proceeding."
        )

    by_frame = {}
    for t, y, x in zip(ts, ys, xs):
        by_frame.setdefault(int(t), []).append((float(y), float(x)))
    return {t: np.array(pts) for t, pts in by_frame.items()}


def build_detector(name):
    if name == "baseline":
        return CellDetector(
            sigma=config.GAUSSIAN_SIGMA,
            threshold_abs=config.DETECTION_THRESHOLD,
            min_distance=config.CELL_RADIUS,
        )
    raise ValueError(
        f"Unknown --detector {name!r}. Only 'baseline' (frozen CellDetector, unmodified) is "
        f"wired up so far -- extend build_detector() here to add e.g. 'adaptive' by importing "
        f"scripts/experiments/exp01_adaptive_threshold/adaptive_detector.py, once that "
        f"comparison is wanted under v2's stricter metric."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--d-max", type=float, default=15.0,
                         help="Distance gate in pixels for a match to count as TP (default 15.0)")
    parser.add_argument("--detector", default="baseline", choices=["baseline"])
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.name in RESERVED_FILENAMES:
        raise SystemExit(
            f"Refusing to write to reserved filename '{out_path.name}' -- that name belongs to "
            f"the frozen Benchmark v1 baseline. Choose a different --out."
        )
    if out_path.exists():
        raise SystemExit(f"Refusing to overwrite existing file {out_path}. Choose a new --out path.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_out_path = out_path.with_name(out_path.stem + "_frames.csv")

    dataset = BioHubDataset(config.DATASET_PATH)
    samples = dataset.train_samples
    if args.limit_samples:
        samples = samples[: args.limit_samples]

    print(f"Benchmark v2: distance-gated matching, d_max={args.d_max}px, detector={args.detector}")
    print(f"Samples: {len(samples)}{' (SMOKE TEST subset)' if args.limit_samples else ''}\n")

    sample_rows = []
    frame_rows = []

    for si, sample in enumerate(samples):
        print(f"[{si + 1}/{len(samples)}] {sample}", flush=True)

        gt_by_frame = load_ground_truth_points(dataset, sample)
        volume = np.asarray(dataset.load_volume(sample))
        detector = build_detector(args.detector)
        raw_detections = detector.detect_volume(volume)

        tp_total = fp_total = fn_total = 0
        n_gt_total = n_pred_total = 0
        all_tp_distances = []

        for t in range(volume.shape[0]):
            gt_pts = gt_by_frame.get(t, np.empty((0, 2)))
            pred_pts = raw_detections[t]

            result = gated_match(gt_pts, pred_pts, args.d_max)

            tp_total += result["tp"]
            fp_total += result["fp"]
            fn_total += result["fn"]
            n_gt_total += len(gt_pts)
            n_pred_total += len(pred_pts)
            all_tp_distances.extend(result["tp_distances"].tolist())

            f_precision, f_recall, f_f1 = precision_recall_f1(result["tp"], result["fp"], result["fn"])
            frame_rows.append({
                "sample": sample, "frame": t,
                "n_gt": len(gt_pts), "n_pred": len(pred_pts),
                "tp": result["tp"], "fp": result["fp"], "fn": result["fn"],
                "precision": f_precision, "recall": f_recall, "f1": f_f1,
                "mean_error_px_tp_only": float(np.mean(result["tp_distances"])) if result["tp"] else float("nan"),
            })

        d = np.array(all_tp_distances)
        precision, recall, f1 = precision_recall_f1(tp_total, fp_total, fn_total)

        sample_rows.append({
            "sample": sample,
            "n_frames": volume.shape[0],
            "d_max_px": args.d_max,
            "n_gt_total": n_gt_total,
            "n_pred_total": n_pred_total,
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_error_px_tp_only": float(np.mean(d)) if len(d) else float("nan"),
            "median_error_px_tp_only": float(np.median(d)) if len(d) else float("nan"),
            "frac_le_5px_tp_only": float(np.mean(d <= 5)) if len(d) else float("nan"),
            "frac_le_10px_tp_only": float(np.mean(d <= 10)) if len(d) else float("nan"),
            "frac_le_15px_tp_only": float(np.mean(d <= 15)) if len(d) else float("nan"),
        })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)
    print(f"\nWrote {len(sample_rows)} sample rows to {out_path}")

    with open(frames_out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frame_rows)
    print(f"Wrote {len(frame_rows)} frame rows to {frames_out_path}")

    print(
        "\nThis is Benchmark v2 output -- a different scoring methodology (distance-gated), "
        "NOT a replacement for results/benchmark_baseline_v1.csv. Both should be reported "
        "side by side, not one in place of the other, since they answer different questions "
        "(v1: how far off are matched points on average; v2: how many real cells were actually "
        "found vs. missed vs. hallucinated)."
    )


if __name__ == "__main__":
    main()
