"""
Runs LabOS against a real, public Cell Tracking Challenge (CTC) dataset — this is v0.7.0's
"run on real public datasets" objective, made concrete and runnable.

What this script does:
  1. Loads a real CTC dataset's ground truth (scripts/load_ctc_ground_truth.py) and raw image
     sequence, from a folder you've downloaded from https://celltrackingchallenge.net/.
  2. Runs LabOS's real detector + HungarianTracker + LineageBuilder on it, timed for real.
  3. Scores the result against the real ground truth (src/benchmark.py::score_method).
  4. Optionally scores a real TrackMate XML export against the SAME ground truth, as a baseline.
  5. Records every row into LabOS's own benchmark_runs table (mvp/experiments.py) — this is
     what makes the results show up on the in-app Performance page (v0.7.0 Epic 2), not just
     print to a terminal once and disappear.

What this script does NOT do: download the dataset for you (celltrackingchallenge.net requires
a browser and has no API this environment could script against; also true offline-sandbox
constraints — see RUNBOOK.md's CTC step for exact download links), or run TrackMate for you
(same reasoning as scripts/run_trackmate_comparison.py — that's a Fiji/Java plugin).

Usage:
    python scripts/run_ctc_benchmark.py \\
        --dataset-dir path/to/Fluo-N2DL-HeLa \\
        --sequence 01 \\
        --trackmate-xml path/to/your_export.xml \\
        --trackmate-runtime-seconds 340.0

--dataset-dir should be the folder you get from unzipping a CTC training-dataset archive — it
must contain both "<sequence>/" (raw images) and "<sequence>_GT/TRA/" (ground truth).
--trackmate-xml and --trackmate-runtime-seconds are optional; omit them to record LabOS's own
numbers only.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.detector import CellDetector
from src.tracker import HungarianTracker
from src.benchmark import run_method, score_method, score_external_method, format_results_table, load_trackmate_xml
from scripts.load_ctc_ground_truth import (
    load_ctc_ground_truth,
    load_ctc_image_sequence,
    load_trackmate_ctc,
)
import mvp.experiments as experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, help="Folder containing <sequence>/ and <sequence>_GT/TRA/")
    parser.add_argument("--sequence", default="01", help="CTC sequence number, usually '01' or '02'")
    parser.add_argument("--trackmate-xml", default=None)
    parser.add_argument("--trackmate-runtime-seconds", type=float, default=None)
    parser.add_argument("--max-distance", type=float, default=15.0,
                         help="Node-matching tolerance in pixels for scoring (CTC cells are often "
                              "larger/more spread out than the Kaggle BioHub dataset's — start "
                              "generous and tighten once you've eyeballed the results)")
    parser.add_argument("--frame-limit", type=int, default=None,
                         help="Process only the first N frames — useful for a fast first pass "
                              "on a large CTC sequence before committing to the full run")
    parser.add_argument("--skip-db", action="store_true",
                         help="Print the table but don't record into benchmark_runs (e.g. for a "
                              "throwaway test run you don't want cluttering the Performance page)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    dataset_name = f"{dataset_dir.name} ({args.sequence}, CTC)"
    gt_tra_dir = dataset_dir / f"{args.sequence}_GT" / "TRA"
    sequence_dir = dataset_dir / args.sequence

    print(f"Dataset: {dataset_name}")
    print(f"Loading ground truth from {gt_tra_dir}...")
    gt_nodes, gt_edges = load_ctc_ground_truth(gt_tra_dir)
    from collections import Counter
    n_gt_divisions = sum(1 for count in Counter(s for s, _ in gt_edges).values() if count >= 2)
    print(f"  {len(gt_nodes)} ground-truth nodes, {len(gt_edges)} ground-truth edges, "
          f"{n_gt_divisions} division(s)")

    print(f"Loading raw image sequence from {sequence_dir}...")
    volume = load_ctc_image_sequence(sequence_dir)
    if args.frame_limit:
        volume = volume[: args.frame_limit]
        gt_nodes = [n for n in gt_nodes if n["t"] < args.frame_limit]
        valid_ids = {n["node_id"] for n in gt_nodes}
        gt_edges = [(s, t) for s, t in gt_edges if s in valid_ids and t in valid_ids]
    print(f"  volume shape (T, Z, Y, X): {volume.shape}")

    print("Running LabOS detection + tracking + lineage, timed...")
    detector = CellDetector(sigma=config.GAUSSIAN_SIGMA, threshold_abs=config.DETECTION_THRESHOLD,
                             min_distance=config.CELL_RADIUS)
    raw_detections = detector.detect_volume(volume)
    tracker = HungarianTracker(max_distance=config.TRACKING_MAX_DISTANCE)
    labos_nodes, labos_edges, labos_elapsed, labos_mem = run_method(
        "LabOS", raw_detections, tracker, use_lineage_builder=True,
        division_max_distance=config.DIVISION_MAX_DISTANCE,
    )
    labos_scores = score_method(
    labos_nodes, labos_edges, gt_nodes, gt_edges,
    n_true_nodes_estimate=None,
    max_distance=args.max_distance
)
    labos_row = {"dataset": dataset_name, "method": "LabOS", "runtime_seconds": labos_elapsed,
                 "memory_delta_mb": labos_mem, **labos_scores}

    rows = [labos_row]
    tm_row = None
    if args.trackmate_xml:
        print(f"Loading TrackMate export from {args.trackmate_xml}...")
        tm_nodes, tm_edges = load_trackmate_ctc(args.trackmate_xml)
        tm_row = score_external_method(
             dataset_name, "TrackMate", tm_nodes, tm_edges, gt_nodes, gt_edges,
             n_true_nodes_estimate=None,
             max_distance=args.max_distance,
             runtime_seconds=args.trackmate_runtime_seconds,
             memory_delta_mb=None,
        )
        rows.append(tm_row)

    print()
    print("=" * 78)
    print(f"LabOS on real public data — {dataset_name}")
    print("=" * 78)
    print(format_results_table(rows))
    print(
    "CAVEAT: Official adjusted CTC score is not reported because the dataset does not provide\n"
    "a verified estimated_number_of_nodes value required by the official metric.\n"
    "Raw edge Jaccard is reported as a diagnostic only and should not be presented as the\n"
    "official CTC tracking score.\n"
)

    if not args.skip_db:
        experiments.record_benchmark_run(
            dataset_name, "LabOS", is_baseline=False,
            adjusted_edge_jaccard=labos_row["adjusted_edge_jaccard"],
            division_jaccard=labos_row["division_jaccard"],
            edge_precision=labos_row["edge_precision"], edge_recall=labos_row["edge_recall"],
            node_detection_rate=labos_row["node_detection_rate"],
            runtime_seconds=labos_row["runtime_seconds"], memory_delta_mb=labos_row["memory_delta_mb"],
            notes=f"Real CTC dataset, sequence {args.sequence}" + (f", first {args.frame_limit} frames" if args.frame_limit else ""),
        )
        if tm_row is not None:
            experiments.record_benchmark_run(
                dataset_name, "TrackMate", is_baseline=True,
                adjusted_edge_jaccard=tm_row["adjusted_edge_jaccard"],
                division_jaccard=tm_row["division_jaccard"],
                edge_precision=tm_row["edge_precision"], edge_recall=tm_row["edge_recall"],
                node_detection_rate=tm_row["node_detection_rate"],
                runtime_seconds=tm_row["runtime_seconds"], memory_delta_mb=tm_row["memory_delta_mb"],
                notes="Real TrackMate export, same ground truth",
            )
        print("Recorded into LabOS's own database — see the Performance page (sidebar) in the app.")
    else:
        print("--skip-db set: not recorded. Remove that flag to save this into the Performance page.")


if __name__ == "__main__":
    main()
