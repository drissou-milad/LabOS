"""
Produces the actual "LabOS vs TrackMate" evidence table for the incubator package.

This does NOT run TrackMate for you — that has to happen in Fiji, on your machine, by hand
(there's no way around that; TrackMate is a Java/ImageJ plugin, not something this Python
environment can invoke). What this script does:

  1. Runs LabOS's own detector + HungarianTracker + LineageBuilder on the SAME real dataset,
     timed for real (not synthetic).
  2. Loads your real TrackMate XML export (via src/benchmark.load_trackmate_xml) and scores it
     against the same ground truth, on equal footing.
  3. Prints ONE combined table — runtime, accuracy (precision/recall/adjusted edge Jaccard),
     and division detection (division Jaccard) — plus a checklist for the two things this
     script genuinely cannot measure for you: manual intervention effort and report-generation
     quality, which are workflow observations, not numbers a script can compute.

Prerequisites (see RUNBOOK.md steps 0, 7, 8 for the full detail on each):
  - BIOHUB_DATASET_PATH set to the real competition dataset (not the synthetic examples/ gallery)
  - scripts/inspect_ground_truth.py has been run and load_ground_truth() below matches what it
    printed (this script reuses run_real_benchmark.py's load_ground_truth exactly, unmodified —
    fix it there, not here, if it doesn't match your data)
  - A real TrackMate XML export from Fiji for the SAME sample/volume

Usage:
    python scripts/run_trackmate_comparison.py \\
        --sample SAMPLE_NAME \\
        --trackmate-xml path/to/your_export.xml \\
        --trackmate-runtime-seconds 42.0

`--trackmate-runtime-seconds` is optional but strongly recommended: time your TrackMate run
in Fiji with a stopwatch (wall clock, from "click Track" to "done") and pass it here. Without
it, the table will honestly print "n/a" for TrackMate's runtime rather than a fabricated 0 —
see src/benchmark.py's score_external_method() docstring.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src import config
from src.dataset import BioHubDataset
from src.detector import CellDetector
from src.tracker import HungarianTracker
from src.lineage import LineageBuilder
from src.benchmark import (
    run_method, score_method, score_external_method, format_results_table,
    load_trackmate_xml,
)
from scripts.run_real_benchmark import load_ground_truth  # reuse, don't duplicate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default=None)
    parser.add_argument("--trackmate-xml", required=True,
                         help="Path to your real TrackMate XML model export")
    parser.add_argument("--trackmate-runtime-seconds", type=float, default=None,
                         help="Wall-clock time you measured for the TrackMate run in Fiji (optional)")
    parser.add_argument("--frame-limit", type=int, default=None)
    args = parser.parse_args()

    dataset = BioHubDataset(config.DATASET_PATH)
    sample = args.sample or dataset.train_samples[0]
    print(f"Sample: {sample}")

    print("Loading ground truth...")
    gt_nodes, gt_edges = load_ground_truth(dataset, sample)
    print(f"  {len(gt_nodes)} ground-truth nodes, {len(gt_edges)} ground-truth edges")

    print("Loading volume and running LabOS detection (real, timed)...")
    volume = np.asarray(dataset.load_volume(sample))
    if args.frame_limit:
        volume = volume[: args.frame_limit]

    detector = CellDetector(
        sigma=config.GAUSSIAN_SIGMA,
        threshold_abs=config.DETECTION_THRESHOLD,
        min_distance=config.CELL_RADIUS,
    )
    raw_detections = detector.detect_volume(volume)
    print(f"  {sum(len(d) for d in raw_detections)} raw detections across {len(raw_detections)} frames")

    print("Running LabOS tracking (Hungarian + LineageBuilder), timed...")
    tracker = HungarianTracker(max_distance=config.TRACKING_MAX_DISTANCE)
    labos_nodes, labos_edges, labos_elapsed, labos_mem = run_method(
        "LabOS", raw_detections, tracker, use_lineage_builder=True,
        division_max_distance=config.DIVISION_MAX_DISTANCE,
    )
    labos_scores = score_method(
        labos_nodes, labos_edges, gt_nodes, gt_edges,
        n_true_nodes_estimate=len(gt_nodes), max_distance=7.0,
    )
    labos_row = {
        "dataset": sample, "method": "LabOS", "runtime_seconds": labos_elapsed,
        "memory_delta_mb": labos_mem, **labos_scores,
    }

    print(f"Loading TrackMate export from {args.trackmate_xml}...")
    tm_nodes, tm_edges = load_trackmate_xml(args.trackmate_xml)
    print(f"  {len(tm_nodes)} TrackMate nodes, {len(tm_edges)} TrackMate edges")
    tm_row = score_external_method(
        sample, "TrackMate", tm_nodes, tm_edges, gt_nodes, gt_edges,
        n_true_nodes_estimate=len(gt_nodes), max_distance=7.0,
        runtime_seconds=args.trackmate_runtime_seconds, memory_delta_mb=None,
    )

    print()
    print("=" * 78)
    print("LabOS vs TrackMate — measured comparison")
    print("=" * 78)
    print(format_results_table([labos_row, tm_row]))

    print(
        "CAVEAT (same one run_real_benchmark.py prints): n_true_nodes_estimate is set to the\n"
        "ground-truth node count itself, which will UNDERSTATE any over-prediction penalty if\n"
        "the real ground truth is sparse. Treat adjusted_edge_jaccard as a reasonable first\n"
        "read for BOTH methods, not a final number — the caveat applies equally to both rows,\n"
        "so it doesn't bias the comparison between them, just the absolute numbers.\n"
    )

    print("Two things this table can't measure — fill these in by hand for the evidence package:")
    print("  - Manual intervention: how many parameters/clicks did each method need to get a")
    print("    usable result? (e.g. LabOS: pick a Configuration, click Run. TrackMate: detector")
    print("    settings, tracker settings, gap-closing settings, manual track editing if any.)")
    print("  - Report generation: does each method produce a shareable report on its own, or")
    print("    does someone have to assemble one afterward? (LabOS: automatic PDF via the")
    print("    Reports tab. TrackMate: typically CSV/XML export only, no built-in report.)")


if __name__ == "__main__":
    main()
