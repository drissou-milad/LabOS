"""
Experiment #3: min_distance (NMS radius) sweep, isolated from thresholding.

Tests H1 (detection density / NMS suppression radius) as a driver of the extreme
false-positive over-detection seen in the frozen Benchmark v2 baseline and in Experiments
#1/#2. Single factor varied: min_distance (peak_local_max's NMS radius). sigma and
threshold_abs are held FIXED at the confirmed frozen-baseline values, established via the
provenance investigation in this conversation (CSV creation timestamp 13 Aug 2026 10:50:45;
config.py content confirmed unchanged since 12 Aug 09:44:47 via matching .pyc embedded source
mtime, flags=0 i.e. timestamp-based invalidation, so every import between the edit and the
baseline run necessarily saw the same content):
    threshold_abs = 10
    sigma = 2
    (min_distance = 5 was the frozen baseline's value; this experiment sweeps around it)

USES THE UNMODIFIED src.detector.CellDetector -- no new detection logic. This is the entire
point of Experiment #3's design: CellDetector already accepts sigma/threshold_abs/min_distance
independently, so isolating min_distance requires zero changes to detection code, only a
runner that constructs it with a swept min_distance and everything else fixed.

PROVENANCE FIX (the reason for this whole investigation): sigma and threshold_abs are passed
as EXPLICIT CLI ARGUMENTS with literal defaults (10, 2) -- this script does NOT import
src.config or read config.GAUSSIAN_SIGMA / config.DETECTION_THRESHOLD / config.CELL_RADIUS at
all. It cannot silently drift if config.py is edited later, because it never reads config.py's
detection parameters in the first place. Every output file (CSV rows AND the summary) records
sigma, threshold_abs, min_distance, d_max, a UTC timestamp, the exact command line (sys.argv),
and a SHA-256 hash of src/detector.py's contents at run time (proving CellDetector itself
wasn't modified for this run, and giving a fingerprint to detect if it ever is later) --
so reconstructing "what config produced this file" never again requires forensic .pyc
archaeology.

Does NOT modify: src/detector.py, src/config.py, matching_v2.py, run_benchmark_v2.py,
run_benchmark_v2_adaptive.py, adaptive_detector.py, or any frozen baseline result.

Usage:
    export BIOHUB_DATASET_PATH=/path/to/biohub-cell-tracking-during-development

    # Smoke test (first 3 samples), one min_distance value
    python scripts/benchmark_v2/run_benchmark_v2_min_distance_sweep.py \\
        --out results/exp03_min_distance_sweep/benchmark_v2_mindist5_smoke.csv \\
        --min-distance 5 --limit-samples 3

    # Full run (all ~199 samples), once smoke tests are reviewed
    python scripts/benchmark_v2/run_benchmark_v2_min_distance_sweep.py \\
        --out results/exp03_min_distance_sweep/benchmark_v2_mindist5.csv \\
        --min-distance 5

Outputs (all under results/exp03_min_distance_sweep/ only):
    <out>.csv          per-sample TP/FP/FN/precision/recall/F1/mean+median TP error, PLUS
                        sigma/threshold_abs/min_distance/d_max_px repeated on every row
    <out>_frames.csv   per-frame breakdown
    <out>_summary.txt  dataset-wide aggregate + full provenance block (params, argv, UTC
                        timestamp, detector.py SHA-256)
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

from src.dataset import BioHubDataset
from src.detector import CellDetector  # unmodified, reused as-is

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matching_v2 import gated_match, precision_recall_f1  # unmodified, reused as-is

RESERVED_FILENAMES = {
    "benchmark_baseline_v1.csv",
    "benchmark_v2_baseline_detector.csv",
}
REQUIRED_OUTPUT_PARENT = "exp03_min_distance_sweep"

# Dataset path is the one config.py concern this script still needs -- but only the path,
# never the detection parameters. Read directly from the environment variable rather than
# importing src.config, to keep this script's only dependency on config.py at zero.
import os
DEFAULT_DATASET_PATH = Path(os.environ.get(
    "BIOHUB_DATASET_PATH",
    r"D:\Datasets\BioHub\biohub-cell-tracking-during-development",
))


def load_ground_truth_points(dataset, sample):
    """Confirmed schema (nodes/props/<name>/values), verified on sample 44b6_0113de3b.
    Duplicated here (not imported from run_benchmark_v2.py) so this script has no dependency
    on a file that already produced frozen results."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--min-distance", type=int, required=True,
                         help="The swept variable (NMS radius, peak_local_max's min_distance)")
    parser.add_argument("--sigma", type=float, default=2.0,
                         help="FIXED, matches the confirmed frozen baseline (default 2.0)")
    parser.add_argument("--threshold-abs", type=float, default=10.0,
                         help="FIXED, matches the confirmed frozen baseline (default 10.0)")
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
            f"'{out_path.parent}'. Experiment #3 outputs must stay in their own directory, "
            f"separate from the baseline and from Experiments #1/#2."
        )
    if out_path.exists():
        raise SystemExit(f"Refusing to overwrite existing file {out_path}. Choose a new --out path.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_out_path = out_path.with_name(out_path.stem + "_frames.csv")
    summary_out_path = out_path.with_name(out_path.stem + "_summary.txt")

    detector_py_path = PROJECT_ROOT / "src" / "detector.py"
    detector_sha256 = sha256_of_file(detector_py_path)
    run_timestamp_utc = datetime.now(timezone.utc).isoformat()

    dataset = BioHubDataset(DEFAULT_DATASET_PATH)
    samples = dataset.train_samples
    if args.limit_samples:
        samples = samples[: args.limit_samples]

    print("Experiment #3: min_distance (NMS) sweep, all other parameters fixed by explicit CLI literal")
    print(f"sigma={args.sigma} (FIXED), threshold_abs={args.threshold_abs} (FIXED), "
          f"min_distance={args.min_distance} (SWEPT VARIABLE), d_max={args.d_max}px")
    print(f"src/detector.py SHA-256: {detector_sha256}")
    print(f"Samples: {len(samples)}{' (SMOKE TEST subset)' if args.limit_samples else ''}\n")

    sample_rows = []
    frame_rows = []

    global_tp = global_fp = global_fn = 0
    global_n_gt = global_n_pred = 0
    global_tp_distances = []

    for si, sample in enumerate(samples):
        print(f"[{si + 1}/{len(samples)}] {sample}", flush=True)

        gt_by_frame = load_ground_truth_points(dataset, sample)
        volume = np.asarray(dataset.load_volume(sample))

        detector = CellDetector(
            sigma=args.sigma,
            threshold_abs=args.threshold_abs,
            min_distance=args.min_distance,
        )
        raw_detections = detector.detect_volume(volume)

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
            })

        d = np.array(sample_tp_distances)
        precision, recall, f1 = precision_recall_f1(tp_total, fp_total, fn_total)

        sample_rows.append({
            "sample": sample,
            "n_frames": volume.shape[0],
            "sigma": args.sigma,
            "threshold_abs": args.threshold_abs,
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
        "Experiment #3: min_distance (NMS) sweep -- Benchmark v2 methodology",
        "=" * 70,
        "PROVENANCE (recorded explicitly so this can never again require forensic reconstruction)",
        f"  run_timestamp_utc = {run_timestamp_utc}",
        f"  command_line = {' '.join(sys.argv)}",
        f"  src/detector.py SHA-256 = {detector_sha256}",
        f"  parameters: sigma={args.sigma} (FIXED), threshold_abs={args.threshold_abs} (FIXED), "
        f"min_distance={args.min_distance} (SWEPT), d_max={args.d_max}px",
        f"  detector class: src.detector.CellDetector (unmodified, imported directly)",
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
        f"precision = {g_precision:.7g}",
        f"recall = {g_recall:.7g}",
        f"F1 = {g_f1:.7g}",
        f"mean TP localization error = {float(np.mean(gd)) if len(gd) else float('nan'):.3f} px",
        f"median TP localization error = {float(np.median(gd)) if len(gd) else float('nan'):.3f} px",
    ]
    summary_text = "\n".join(summary_lines)
    summary_out_path.write_text(summary_text)
    print("\n" + summary_text)


if __name__ == "__main__":
    main()
