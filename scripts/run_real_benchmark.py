"""
Runs src/benchmark.py against a REAL training sample's ground truth, instead of the synthetic
demo in the README. This is what turns "we have a scorer" into "we have a number."

RUN scripts/inspect_ground_truth.py FIRST and read its output. The conversion function below
(`load_ground_truth`) is written from partial, secondhand memory of what an earlier notebook
session printed about the graph structure — it's a reasonable starting guess, not a verified
fact. If inspect_ground_truth.py's output doesn't match what this function assumes, fix the
function before trusting anything this script prints.

Usage:
    python scripts/inspect_ground_truth.py          # do this first
    python scripts/run_real_benchmark.py --sample SAMPLE_NAME
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
from src.benchmark import NaiveGreedyTracker, run_benchmark, format_results_table


def load_ground_truth(dataset, sample):
    """
    Converts BioHubDataset.load_graph(sample) into (nodes, edges) in the shape
    src/evaluate.py expects: nodes = list of {"node_id","t","y","x"}, edges = list of
    (source_id, target_id).

    ASSUMPTION (verify against scripts/inspect_ground_truth.py's output before trusting this):
    graph["nodes"] has "ids", "t", "y", "x" arrays (possibly also "z", ignored here since
    src/evaluate.py's node matching is 2D, matching how CellDetector/LineageBuilder work on
    max-projections); graph["edges"] has an "ids" array of shape (N, 2), each row a
    (source_node_id, target_node_id) pair. If the real structure differs — e.g. separate
    "source"/"target" arrays instead of one paired array — this is the function to edit; the
    scoring logic downstream doesn't change.
    """
    graph = dataset.load_graph(sample)

    node_ids = graph["nodes"]["ids"][:]
    ts = graph["nodes"]["t"][:]
    ys = graph["nodes"]["y"][:]
    xs = graph["nodes"]["x"][:]
    nodes = [
        {"node_id": int(node_ids[i]), "t": int(ts[i]), "y": float(ys[i]), "x": float(xs[i])}
        for i in range(len(node_ids))
    ]

    edge_ids = graph["edges"]["ids"][:]
    if edge_ids.ndim == 2 and edge_ids.shape[1] == 2:
        edges = [(int(s), int(t)) for s, t in edge_ids]
    else:
        raise ValueError(
            f"graph['edges']['ids'] has shape {edge_ids.shape}, not the assumed (N, 2). "
            "Run scripts/inspect_ground_truth.py and fix load_ground_truth() in this file "
            "to match the real structure before trusting this script's output."
        )

    return nodes, edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default=None)
    parser.add_argument("--frame-limit", type=int, default=None,
                         help="Limit to the first N frames, for a faster first run")
    args = parser.parse_args()

    dataset = BioHubDataset(config.DATASET_PATH)
    sample = args.sample or dataset.train_samples[0]
    print(f"Sample: {sample}")

    print("Loading ground truth...")
    gt_nodes, gt_edges = load_ground_truth(dataset, sample)
    print(f"  {len(gt_nodes)} ground-truth nodes, {len(gt_edges)} ground-truth edges")

    print("Loading volume and detecting cells (this is the SAME detector output both methods")
    print("below will be scored on — see src/benchmark.py's docstring on why that's the fair way")
    print("to compare tracking methods specifically, not detection quality)...")
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

    # NOTE: this benchmarks TRACKING, not the full pipeline — raw detector output goes straight
    # into both trackers below, skipping CNN filtering. That's intentional (isolates the
    # tracking-algorithm comparison from detection-quality noise) but means these numbers are
    # not directly comparable to a real submission.csv's score, which does include CNN filtering.
    methods = {
        "Naive greedy (baseline)": (NaiveGreedyTracker(max_distance=config.TRACKING_MAX_DISTANCE), False, None),
        "Hungarian + LineageBuilder (ours)": (
            HungarianTracker(max_distance=config.TRACKING_MAX_DISTANCE), True, config.DIVISION_MAX_DISTANCE
        ),
    }

    print("Running benchmark...")
    rows = run_benchmark(
        sample, raw_detections, gt_nodes, gt_edges, methods,
        n_true_nodes_estimate=len(gt_nodes),  # a real estimate would come from the competition, not GT count itself — see caveat below
        max_distance=7.0,  # matches the competition's published node-matching tolerance
    )

    print()
    print(format_results_table(rows))
    print()
    print("CAVEAT: n_true_nodes_estimate is set to the ground-truth node count itself here,")
    print("which will UNDERSTATE any over-prediction penalty (adjusted_edge_jaccard) if the")
    print("real ground truth is sparse (doesn't annotate every true cell) — see")
    print("src/evaluate.py's docstring on adjusted_edge_jaccard for what this number is")
    print("actually supposed to be. Treat these scores as a reasonable first read, not final.")


if __name__ == "__main__":
    main()
