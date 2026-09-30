"""
Experiment #4: Otsu adaptive threshold x min_distance interaction.

Tests H4: does moderate NMS-radius adjustment (min_distance=8 or 12) improve F1 on top of
Otsu adaptive thresholding, without reproducing the severe recall collapse Experiment #3
showed under FIXED thresholding at the same min_distance values? Full design doc: this
conversation, message preceding implementation.

Single factor varied: min_distance. threshold_method is fixed to "otsu" (CLI choice
deliberately restricted to just "otsu" -- this experiment does not test percentile/robust
under a min_distance sweep; that would be a different, unapproved experiment). sigma and
threshold_floor are held fixed at the confirmed frozen-baseline values (2, 10).

USES THE UNMODIFIED AdaptiveCellDetector from
scripts/experiments/exp01_adaptive_threshold/adaptive_detector.py -- no new detection logic.
Only min_distance differs between runs of this script.

WHY A NEW FILE, NOT A MODIFICATION OF run_benchmark_v2_adaptive.py: that script reads
sigma/min_distance/threshold_floor live from src.config at import time (confirmed by
inspection before this file was written) -- the same provenance hazard investigated at length
earlier in this project. It also already produced the trusted, cited, full-scale Otsu
reference result (F1=0.1469, min_distance=5). Modifying it risks altering behavior for that
already-completed, already-cited run. This script instead takes sigma / threshold_method /
threshold_floor / min_distance / d_max as EXPLICIT CLI LITERALS ONLY and never imports
src.config for detection parameters, so it cannot silently drift the same way, ever.

PROVENANCE: every CSV row carries sigma/threshold_method/threshold_floor/min_distance/d_max_px
as literal columns. The summary.txt records a UTC timestamp, the exact command line
(sys.argv), and a SHA-256 of adaptive_detector.py (the file actually performing the
thresholding here, unlike Experiment #3 which hashed src/detector.py) at run time.

STATISTICS: the summary reports Wilson 95% confidence intervals for precision and recall.
Per the approved experiment design, these are DIAGNOSTIC ONLY -- this script does not gate,
assert, or make any pass/fail decision based on them. All decision-making happens in the
analysis step after results are reported, not in this script. F1's confidence interval is NOT
computed (no simple closed form exists for a ratio of two correlated Wilson-interval
quantities without bootstrapping, which is out of scope here) -- this is stated explicitly in
the summary output rather than silently omitted.

Does NOT modify: src/detector.py, src/config.py, matching_v2.py, run_benchmark_v2.py,
run_benchmark_v2_adaptive.py, run_benchmark_v2_min_distance_sweep.py, adaptive_detector.py,
or any frozen baseline / prior experiment result.

Usage:
    export BIOHUB_DATASET_PATH=/path/to/biohub-cell-tracking-during-development

    # Smoke tests (first 3 samples), one per min_distance value
    python scripts/benchmark_v2/run_benchmark_v2_otsu_min_distance.py \\
        --out results/exp04_otsu_min_distance/benchmark_v2_otsu_md5_smoke.csv \\
        --min-distance 5 --limit-samples 3
    python scripts/benchmark_v2/run_benchmark_v2_otsu_min_distance.py \\
        --out results/exp04_otsu_min_distance/benchmark_v2_otsu_md8_smoke.csv \\
        --min-distance 8 --limit-samples 3
    python scripts/benchmark_v2/run_benchmark_v2_otsu_min_distance.py \\
        --out results/exp04_otsu_min_distance/benchmark_v2_otsu_md12_smoke.csv \\
        --min-distance 12 --limit-samples 3

Outputs (all under results/exp04_otsu_min_distance/ only):
    <out>.csv          per-sample TP/FP/FN/precision/recall/F1/mean+median TP error/threshold
                        stats, PLUS sigma/threshold_method/threshold_floor/min_distance/d_max_px
                        repeated on every row
    <out>_frames.csv   per-frame breakdown
    <out>_summary.txt  dataset-wide aggregate, Wilson 95% CIs (diagnostic only), and full
                        provenance block
"""

import argparse
import csv
import hashlib
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.dataset import BioHubDataset  # unmodified; only used to load data, not parameters

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matching_v2 import gated_match, precision_recall_f1  # unmodified, reused as-is

ADAPTIVE_DETECTOR_DIR = (
    PROJECT_ROOT / "scripts" / "experiments" / "exp01_adaptive_threshold"
)
sys.path.insert(0, str(ADAPTIVE_DETECTOR_DIR))
from adaptive_detector import AdaptiveCellDetector  # unmodified, reused as-is

RESERVED_FILENAMES = {
    "benchmark_baseline_v1.csv",
    "benchmark_v2_baseline_detector.csv",
}
REQUIRED_OUTPUT_PARENT = "exp04_otsu_min_distance"

import os
DEFAULT_DATASET_PATH = Path(os.environ.get(
    "BIOHUB_DATASET_PATH",
    r"D:\Datasets\BioHub\biohub-cell-tracking-during-development",
))


def load_ground_truth_points(dataset, sample):
    """Confirmed schema (nodes/props/<name>/values), verified on sample 44b6_0113de3b.
    Duplicated here (not imported), same isolation rationale as Experiment #3's copy."""
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


def sha256_of_file(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception as e:
        return f"<could not hash: {e}>"


def wilson_ci(k, n, z=1.959963985):
    """Wilson score interval for a binomial proportion k/n. Returns (lo, hi), or (nan, nan)
    if n==0. DIAGNOSTIC ONLY -- see module docstring: not used for any pass/fail decision in
    this script."""
    if n == 0:
        return float("nan"), float("nan")
    p_hat = k / n
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--min-distance", type=int, required=True,
                         help="The swept variable (NMS radius). Approved values: 5, 8, 12.")
    parser.add_argument("--sigma", type=float, default=2.0,
                         help="FIXED, matches confirmed frozen baseline / Experiment #1")
    parser.add_argument("--threshold-floor", type=float, default=10.0,
                         help="FIXED, matches confirmed frozen baseline / Experiment #1")
    parser.add_argument("--threshold-method", choices=["otsu"], default="otsu",
                         help="Restricted to 'otsu' -- this experiment does not test other "
                              "threshold methods under a min_distance sweep")
    parser.add_argument("--d-max", type=float, default=15.0)
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
            f"'{out_path.parent}'. This categorically prevents collision with "
            f"results/exp01_adaptive_threshold/, which holds the trusted full-scale Otsu "
            f"reference result."
        )
    if out_path.exists():
        raise SystemExit(f"Refusing to overwrite existing file {out_path}. Choose a new --out path.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_out_path = out_path.with_name(out_path.stem + "_frames.csv")
    summary_out_path = out_path.with_name(out_path.stem + "_summary.txt")

    adaptive_detector_py_path = ADAPTIVE_DETECTOR_DIR / "adaptive_detector.py"
    detector_sha256 = sha256_of_file(adaptive_detector_py_path)
    run_timestamp_utc = datetime.now(timezone.utc).isoformat()

    dataset = BioHubDataset(DEFAULT_DATASET_PATH)
    samples = dataset.train_samples
    if args.limit_samples:
        samples = samples[: args.limit_samples]

    print("Experiment #4: Otsu x min_distance interaction, all other parameters fixed by "
          "explicit CLI literal")
    print(f"sigma={args.sigma} (FIXED), threshold_method={args.threshold_method} (FIXED), "
          f"threshold_floor={args.threshold_floor} (FIXED), "
          f"min_distance={args.min_distance} (SWEPT VARIABLE), d_max={args.d_max}px")
    print(f"adaptive_detector.py SHA-256: {detector_sha256}")
    print(f"Samples: {len(samples)}{' (SMOKE TEST subset)' if args.limit_samples else ''}\n")

    sample_rows = []
    frame_rows = []

    global_tp = global_fp = global_fn = 0
    global_n_gt = global_n_pred = 0
    global_tp_distances = []
    global_thresholds_used = []

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
        thresholds_used = detector.last_thresholds_used
        global_thresholds_used.extend(thresholds_used)

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
            "sigma": args.sigma,
            "threshold_method": args.threshold_method,
            "threshold_floor": args.threshold_floor,
            "min_distance": args.min_distance,
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

    precision_ci_lo, precision_ci_hi = wilson_ci(global_tp, global_n_pred)
    recall_ci_lo, recall_ci_hi = wilson_ci(global_tp, global_n_gt)

    summary_lines = [
        "Experiment #4: Otsu x min_distance interaction -- Benchmark v2 methodology",
        "=" * 70,
        "PROVENANCE",
        f"  run_timestamp_utc = {run_timestamp_utc}",
        f"  command_line = {' '.join(sys.argv)}",
        f"  adaptive_detector.py SHA-256 = {detector_sha256}",
        f"  parameters: sigma={args.sigma} (FIXED), threshold_method={args.threshold_method} "
        f"(FIXED), threshold_floor={args.threshold_floor} (FIXED), "
        f"min_distance={args.min_distance} (SWEPT), d_max={args.d_max}px",
        f"  detector class: AdaptiveCellDetector "
        f"(scripts/experiments/exp01_adaptive_threshold/adaptive_detector.py, unmodified)",
        f"  dataset_path = {DEFAULT_DATASET_PATH}",
        "=" * 70,
        "",
        f"Samples: {len(samples)}, Frames: {sum(r['n_frames'] for r in sample_rows)}",
        "",
        f"GT = {global_n_gt}",
        f"predictions = {global_n_pred}",
        f"TP = {global_tp}",
        f"FP = {global_fp}",
        f"FN = {global_fn}",
        f"precision = {g_precision:.7g}  (Wilson 95% CI: [{precision_ci_lo:.5f}, {precision_ci_hi:.5f}] -- diagnostic only, NOT a gating criterion)",
        f"recall = {g_recall:.7g}  (Wilson 95% CI: [{recall_ci_lo:.5f}, {recall_ci_hi:.5f}] -- diagnostic only, NOT a gating criterion)",
        f"F1 = {g_f1:.7g}  (no closed-form CI computed -- F1 is a nonlinear function of two "
        f"correlated proportions; would require bootstrapping, out of scope here)",
        f"mean TP localization error = {float(np.mean(gd)) if len(gd) else float('nan'):.3f} px",
        f"median TP localization error = {float(np.median(gd)) if len(gd) else float('nan'):.3f} px",
        f"mean threshold used (Otsu) = {float(np.mean(global_thresholds_used)) if global_thresholds_used else float('nan'):.3f}",
        f"threshold std across frames = {float(np.std(global_thresholds_used)) if global_thresholds_used else float('nan'):.3f}",
    ]
    summary_text = "\n".join(summary_lines)
    summary_out_path.write_text(summary_text)
    print("\n" + summary_text)


if __name__ == "__main__":
    main()
