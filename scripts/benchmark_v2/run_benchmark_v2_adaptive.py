"""
Benchmark v2 (distance-gated, d_max=15px) applied to Experiment #1's AdaptiveCellDetector.

This is a NEW, separate script. It does NOT modify:
  - scripts/benchmark_v2/run_benchmark_v2.py   (already used to produce the frozen v2 baseline)
  - scripts/benchmark_v2/matching_v2.py         (imported and reused unmodified)
  - src/detector.py, src/config.py, src/dataset.py
  - any existing results/ file

It reuses matching_v2.gated_match / precision_recall_f1 UNMODIFIED, and its own copy of the
CONFIRMED-CORRECT geff loader (nodes/props/<name>/values schema, verified against the real
dataset on sample 44b6_0113de3b — see scripts/benchmark_v2/run_benchmark_v2.py's
load_ground_truth_points for the same logic). It does not import that function from
run_benchmark_v2.py, to avoid creating a dependency on a script that already produced frozen
results — this script owns its own copy so it can never accidentally be affected by, or affect,
that file.

Controlled-experiment discipline (per the failure-analysis protocol): sigma and CELL_RADIUS
(NMS min_distance) are read from config.py and left EXACTLY as the frozen baseline uses them.
Only the intensity-threshold mechanism changes (AdaptiveCellDetector, Otsu by default). This
isolates the effect of adaptive thresholding from any change to detection scale or NMS radius.

Experiment #2 (--threshold-method robust): adds a background-relative statistic,
T = median(I) + k*MAD(I), as an alternative to Otsu -- see adaptive_detector.py's docstring for
the motivation (Otsu threshold showed a negative correlation with precision/F1 in Experiment #1
smoke tests). Controlled via --threshold-robust-k (default 3.0). Same sigma/min_distance/d_max/
matching/dataset-loading discipline as above; only the threshold statistic differs.

Usage:
    export BIOHUB_DATASET_PATH=/path/to/biohub-cell-tracking-during-development

    # Smoke test
    python scripts/benchmark_v2/run_benchmark_v2_adaptive.py \\
        --out results/exp01_adaptive_threshold/benchmark_v2_adaptive_smoke.csv \\
        --limit-samples 3

    # Full run, all 199 samples
    python scripts/benchmark_v2/run_benchmark_v2_adaptive.py \\
        --out results/exp01_adaptive_threshold/benchmark_v2_adaptive.csv

Outputs (all under results/exp01_adaptive_threshold/ only -- never results/benchmark_v2/):
    <out>.csv                 per-sample TP/FP/FN/precision/recall/F1/mean+median TP error
    <out>_frames.csv          per-frame breakdown
    <out>_summary.txt         dataset-wide aggregate (GT, predictions, TP, FP, FN, precision,
                               recall, F1, mean TP error, median TP error) -- same fields, same
                               order, as the frozen baseline report, computed from the TRUE
                               pooled TP-distance array (not reconstructed from per-sample
                               summary stats), so mean and median are both exact.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matching_v2 import gated_match, precision_recall_f1  # unmodified, reused as-is

ADAPTIVE_DETECTOR_DIR = (
    PROJECT_ROOT / "scripts" / "experiments" / "exp01_adaptive_threshold"
)
sys.path.insert(0, str(ADAPTIVE_DETECTOR_DIR))
from adaptive_detector import AdaptiveCellDetector  # unmodified, reused as-is

# Filenames this script must never write to, regardless of --out.
RESERVED_FILENAMES = {
    "benchmark_baseline_v1.csv",
    "benchmark_v2_baseline_detector.csv",  # the frozen Benchmark v2 baseline, per the v2 README
}
REQUIRED_OUTPUT_PARENT = "exp01_adaptive_threshold"  # enforce the requested separate-output rule


def load_ground_truth_points(dataset, sample):
    """Confirmed schema (nodes/props/<name>/values), verified on sample 44b6_0113de3b.
    Identical logic to the already-fixed run_benchmark_v2.py, duplicated here on purpose so
    this script has no import dependency on that already-used, already-frozen-baseline file."""
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
            f"scoring against untrustworthy ground truth."
        )

    by_frame = {}
    for t, y, x in zip(ts, ys, xs):
        by_frame.setdefault(int(t), []).append((float(y), float(x)))
    return {t: np.array(pts) for t, pts in by_frame.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--d-max", type=float, default=15.0)
    parser.add_argument("--threshold-method", choices=["otsu", "percentile", "robust"], default="otsu")
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--threshold-robust-k", type=float, default=3.0,
                         help="k in T = median(I) + k*MAD(I), only used when "
                              "--threshold-method robust (Experiment #2)")
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.name in RESERVED_FILENAMES:
        raise SystemExit(
            f"Refusing to write to reserved filename '{out_path.name}' -- belongs to a frozen "
            f"baseline. Choose a different --out."
        )
    if out_path.parent.name != REQUIRED_OUTPUT_PARENT:
        raise SystemExit(
            f"Refusing to write outside results/{REQUIRED_OUTPUT_PARENT}/ -- got "
            f"'{out_path.parent}'. Per protocol, Experiment #1 outputs must stay separate from "
            f"any baseline results directory."
        )
    if out_path.exists():
        raise SystemExit(f"Refusing to overwrite existing file {out_path}. Choose a new --out path.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_out_path = out_path.with_name(out_path.stem + "_frames.csv")
    summary_out_path = out_path.with_name(out_path.stem + "_summary.txt")

    dataset = BioHubDataset(config.DATASET_PATH)
    samples = dataset.train_samples
    if args.limit_samples:
        samples = samples[: args.limit_samples]

    robust_k_note = f", robust_k={args.threshold_robust_k}" if args.threshold_method == "robust" else ""
    print(f"Experiment under Benchmark v2 methodology: threshold_method={args.threshold_method}"
          f"{robust_k_note}, d_max={args.d_max}px")
    print(f"sigma={config.GAUSSIAN_SIGMA} (frozen, unchanged), "
          f"min_distance={config.CELL_RADIUS} (frozen, unchanged), "
          f"threshold_floor={config.DETECTION_THRESHOLD} (frozen baseline's fixed value, "
          f"used only as a safety floor)")
    print(f"Samples: {len(samples)}{' (SMOKE TEST subset)' if args.limit_samples else ''}\n")

    sample_rows = []
    frame_rows = []

    # Dataset-wide pooled accumulators -- these give the TRUE global aggregate, not a
    # reconstruction from per-sample summary stats.
    global_tp = global_fp = global_fn = 0
    global_n_gt = global_n_pred = 0
    global_tp_distances = []

    for si, sample in enumerate(samples):
        print(f"[{si + 1}/{len(samples)}] {sample}", flush=True)

        gt_by_frame = load_ground_truth_points(dataset, sample)
        volume = np.asarray(dataset.load_volume(sample))

        detector = AdaptiveCellDetector(
            sigma=config.GAUSSIAN_SIGMA,
            min_distance=config.CELL_RADIUS,
            threshold_method=args.threshold_method,
            threshold_percentile=args.threshold_percentile,
            threshold_floor=config.DETECTION_THRESHOLD,
            robust_k=args.threshold_robust_k,
        )
        raw_detections = detector.detect_volume(volume)
        thresholds_used = detector.last_thresholds_used

        tp_total = fp_total = fn_total = 0
        n_gt_total = n_pred_total = 0
        sample_tp_distances = []

        for t in range(volume.shape[0]):
            gt_pts = gt_by_frame.get(t, np.empty((0, 2)))
            pred_pts = raw_detections[t]

            result = gated_match(gt_pts, pred_pts, args.d_max)

            tp_total += result["tp"]
            fp_total += result["fp"]
            fn_total += result["fn"]
            n_gt_total += len(gt_pts)
            n_pred_total += len(pred_pts)
            sample_tp_distances.extend(result["tp_distances"].tolist())

            f_p, f_r, f_f1 = precision_recall_f1(result["tp"], result["fp"], result["fn"])
            frame_rows.append({
                "sample": sample, "frame": t,
                "n_gt": len(gt_pts), "n_pred": len(pred_pts),
                "tp": result["tp"], "fp": result["fp"], "fn": result["fn"],
                "precision": f_p, "recall": f_r, "f1": f_f1,
                "mean_error_px_tp_only": float(np.mean(result["tp_distances"])) if result["tp"] else float("nan"),
                "threshold_used": thresholds_used[t],
            })

        d = np.array(sample_tp_distances)
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
            "mean_threshold_used": float(np.mean(thresholds_used)) if thresholds_used else float("nan"),
            "threshold_std_across_frames": float(np.std(thresholds_used)) if thresholds_used else float("nan"),
        })

        global_tp += tp_total
        global_fp += fp_total
        global_fn += fn_total
        global_n_gt += n_gt_total
        global_n_pred += n_pred_total
        global_tp_distances.extend(sample_tp_distances)

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

    g_precision, g_recall, g_f1 = precision_recall_f1(global_tp, global_fp, global_fn)
    gd = np.array(global_tp_distances)
    summary_lines = [
        f"Experiment (AdaptiveCellDetector, threshold_method={args.threshold_method}{robust_k_note}) "
        f"-- Benchmark v2, d_max={args.d_max}px",
        f"Samples: {len(samples)}, Frames: {sum(r['n_frames'] for r in sample_rows)}",
        "",
        f"GT = {global_n_gt}",
        f"predictions = {global_n_pred}",
        f"TP = {global_tp}",
        f"FP = {global_fp}",
        f"FN = {global_fn}",
        f"precision = {g_precision:.7g}",
        f"recall = {g_recall:.7g}",
        f"F1 = {g_f1:.7g}",
        f"mean TP localization error = {float(np.mean(gd)) if len(gd) else float('nan'):.3f} px",
        f"median TP localization error = {float(np.median(gd)) if len(gd) else float('nan'):.3f} px",
    ]
    summary_text = "\n".join(summary_lines)
    summary_out_path.write_text(summary_text)
    print("\n" + summary_text)

    print(
        f"\nNext step: python scripts/benchmark_v2/compare_v2_aggregate.py "
        f"--baseline-csv results/benchmark_v2/benchmark_v2_baseline_detector.csv "
        f"--candidate-csv {out_path} "
        f"--out-dir results/exp01_adaptive_threshold/comparison_v2"
    )


if __name__ == "__main__":
    main()
