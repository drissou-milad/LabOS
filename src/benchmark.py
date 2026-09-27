"""
Benchmark harness: runs one or more tracking methods on the same input, scores each against
ground truth via src/evaluate.py, times each, and formats a comparison table.

Two important honesty notes:

1. "TrackMate" here means: this module can *import* TrackMate's own exported results (its XML
   model export) into the same node/edge shape src/evaluate.py expects, so a real TrackMate run
   done in Fiji can be scored on equal footing against this project's pipeline. It does NOT run
   TrackMate itself — there's no Fiji/ImageJ in this environment to run it with, and
   `load_trackmate_xml()` below is written from TrackMate's documented XML schema, not tested
   against a real exported file. Verify it against an actual TrackMate export before trusting
   its numbers in anything that matters.

2. The "naive baseline" (NaiveGreedyTracker) is a deliberately simple greedy nearest-neighbor
   tracker with no division handling — not TrackMate, not a strawman either, just the honest
   "what if you did the simplest possible thing" comparison point that's always available
   without external tools.
"""

import time
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import psutil
from scipy.spatial.distance import cdist

from src.tracker import HungarianTracker
from src.lineage import LineageBuilder
from src.evaluate import (
    match_nodes,
    edge_confusion,
    edge_jaccard,
    adjusted_edge_jaccard,
    division_confusion,
    division_jaccard,
    final_score,
)

_process = psutil.Process()


class NaiveGreedyTracker:
    """The simplest possible baseline: greedy nearest-neighbor matching, first-come-first-served
    (not globally optimal like HungarianTracker, and no division handling at all). Exists purely
    as a comparison point — "how much does the actual algorithm buy you over doing the obvious
    naive thing."""

    def __init__(self, max_distance):
        self.max_distance = max_distance

    def match(self, previous, current):
        previous = np.asarray(previous)
        current = np.asarray(current)
        if len(previous) == 0 or len(current) == 0:
            return []

        dist_matrix = cdist(previous, current)
        matches = []
        used_curr = set()
        # Greedy: for each previous point (in order), grab the nearest still-unused current
        # point within range. No global optimality, unlike HungarianTracker.
        for i in range(len(previous)):
            candidates = [
                (dist_matrix[i, j], j) for j in range(len(current))
                if j not in used_curr and dist_matrix[i, j] <= self.max_distance
            ]
            if candidates:
                _, j = min(candidates)
                matches.append({"previous": i, "current": j, "distance": dist_matrix[i, j]})
                used_curr.add(j)
        return matches


def run_method(name, frames, tracker, use_lineage_builder, division_max_distance=None):
    """
    frames: list of Nx2 (y, x) arrays, one per timepoint (detector + CNN filtering already
        applied — this harness benchmarks tracking, not detection).
    tracker: an object with .match(previous, current) -> list of {"previous","current"} dicts
        (HungarianTracker or NaiveGreedyTracker both satisfy this).
    use_lineage_builder: if True, runs the full LineageBuilder (division detection + track
        IDs) on top of `tracker`; if False, does plain frame-to-frame linking only (no division
        detection) — this is how NaiveGreedyTracker gets benchmarked fairly, since it was never
        designed to catch divisions.

    Returns (nodes, edges, elapsed_seconds, peak_memory_delta_mb).

    Memory caveat (same as scripts/profile_memory.py): RSS delta measured via psutil, which is
    honest about what actually happened but noisy for small inputs — Python/NumPy's own
    allocator behavior can dominate the signal at small scale. Trust the runtime comparison
    more than the memory comparison unless you're running this on realistically large data.
    """
    import gc
    gc.collect()
    mem_before = _process.memory_info().rss / (1024 * 1024)
    start = time.perf_counter()

    if use_lineage_builder:
        builder = LineageBuilder(tracker, division_max_distance=division_max_distance)
        result = builder.build(frames)
        nodes = result.to_dataframe()[["node_id", "t", "y", "x"]].to_dict("records")
        edges = result.edges
    else:
        # Plain linear linking, no track/division bookkeeping — just enough structure to
        # evaluate with src/evaluate.py.
        nodes = []
        edges = []
        node_id = 0
        prev_coords = np.asarray(frames[0]) if len(frames[0]) else np.empty((0, 2))
        prev_ids = []
        for y, x in prev_coords:
            nodes.append({"node_id": node_id, "t": 0, "y": float(y), "x": float(x)})
            prev_ids.append(node_id)
            node_id += 1
        for t in range(1, len(frames)):
            curr_coords = np.asarray(frames[t]) if len(frames[t]) else np.empty((0, 2))
            curr_ids = [None] * len(curr_coords)
            for m in tracker.match(prev_coords, curr_coords):
                y, x = curr_coords[m["current"]]
                nodes.append({"node_id": node_id, "t": t, "y": float(y), "x": float(x)})
                edges.append((prev_ids[m["previous"]], node_id))
                curr_ids[m["current"]] = node_id
                node_id += 1
            for j in range(len(curr_coords)):
                if curr_ids[j] is None:
                    y, x = curr_coords[j]
                    nodes.append({"node_id": node_id, "t": t, "y": float(y), "x": float(x)})
                    curr_ids[j] = node_id
                    node_id += 1
            prev_coords = curr_coords
            prev_ids = curr_ids

    elapsed = time.perf_counter() - start
    gc.collect()
    mem_after = _process.memory_info().rss / (1024 * 1024)
    return nodes, edges, elapsed, mem_after - mem_before


def score_method(pred_nodes, pred_edges, gt_nodes, gt_edges, n_true_nodes_estimate=None,
                 max_distance=7.0):
    """
    Computes benchmark diagnostics.

    Official adjusted CTC score requires a verified coarse estimate of the total
    true node count (including unannotated cells). If that value is unavailable,
    adjusted_edge_jaccard and final_score are reported as None rather than
    producing a misleading score.
    """
    nm = match_nodes(pred_nodes, gt_nodes, max_distance=max_distance)

    tp, fp, fn = edge_confusion(
    pred_edges,
    gt_edges,
    nm,
    pred_nodes=pred_nodes,
    gt_nodes=gt_nodes,
    )
    j = edge_jaccard(tp, fp, fn)

    if n_true_nodes_estimate is not None and n_true_nodes_estimate > 0:
        aej = adjusted_edge_jaccard(
            j,
            n_pred_nodes=len(pred_nodes),
            n_true_nodes_estimate=n_true_nodes_estimate,
        )
        fs = final_score(aej, division_jaccard(*division_confusion(pred_edges, gt_edges, nm)))
    else:
        aej = None
        fs = None

    dtp, dfp, dfn = division_confusion(pred_edges, gt_edges, nm)
    dj = division_jaccard(dtp, dfp, dfn)

    edge_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    edge_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    matched_gt_node_ids = set(nm.values())
    node_detection_rate = (
        len(matched_gt_node_ids) / len(gt_nodes)
        if gt_nodes else 0.0
    )

    return {
        "raw_edge_jaccard": j,
        "adjusted_edge_jaccard": aej,
        "division_jaccard": dj,
        "final_score": fs,
        "edge_precision": edge_precision,
        "edge_recall": edge_recall,
        "node_detection_rate": node_detection_rate,
        "edge_tp": tp,
        "edge_fp": fp,
        "edge_fn": fn,
        "division_tp": dtp,
        "division_fp": dfp,
        "division_fn": dfn,
    }


def run_benchmark(dataset_name, frames, gt_nodes, gt_edges, methods, n_true_nodes_estimate,
                   max_distance=7.0):
    """
    methods: dict of name -> (tracker, use_lineage_builder, division_max_distance) tuples.
    Returns a list of result rows (dicts), one per method, ready for format_results_table()
    or plot_benchmark_dashboard().
    """
    rows = []
    for name, (tracker, use_lineage_builder, division_max_distance) in methods.items():
        pred_nodes, pred_edges, elapsed, memory_delta_mb = run_method(
            name, frames, tracker, use_lineage_builder, division_max_distance
        )
        scores = score_method(pred_nodes, pred_edges, gt_nodes, gt_edges,
                               n_true_nodes_estimate, max_distance=max_distance)
        rows.append({
            "dataset": dataset_name,
            "method": name,
            "runtime_seconds": elapsed,
            "memory_delta_mb": memory_delta_mb,
            **scores,
        })
    return rows


def score_external_method(dataset_name, method_name, pred_nodes, pred_edges, gt_nodes, gt_edges,
                           n_true_nodes_estimate, max_distance=7.0,
                           runtime_seconds=None, memory_delta_mb=None):
    """
    For a result computed outside this harness — most likely `load_trackmate_xml()`'s output,
    or any other tool's export — so it can appear as a row in the same benchmark dashboard as
    `run_benchmark()`'s methods, without pretending this harness measured its runtime/memory
    (which it didn't run, so it can't have timed). Pass runtime_seconds/memory_delta_mb only if
    you have real numbers for them (e.g. from Fiji's own timing); they display as "n/a"
    otherwise rather than a fabricated 0.
    """
    scores = score_method(pred_nodes, pred_edges, gt_nodes, gt_edges,
                           n_true_nodes_estimate, max_distance=max_distance)
    return {
        "dataset": dataset_name,
        "method": method_name,
        "runtime_seconds": runtime_seconds,
        "memory_delta_mb": memory_delta_mb,
        **scores,
    }



def format_results_table(rows):
        """Markdown table for benchmark diagnostics and available official metrics."""
        header = (
            "| Dataset | Method | Runtime | Memory | Node Detection Rate | Precision | Recall | "
            "Raw Edge Jaccard | Adjusted Edge Jaccard | Division Jaccard | Tracking Score |\n"
        )
        header += "|---|---|---|---|---|---|---|---|---|---|---|\n"

        lines = [header]

        for r in rows:
            runtime_str = (
                f"{r['runtime_seconds']:.3f}s"
                if r.get("runtime_seconds") is not None else "n/a"
            )

            memory_str = (
                f"{r['memory_delta_mb']:+.2f} MB"
                if r.get("memory_delta_mb") is not None else "n/a"
            )
            aej_str = (
                f"{r['adjusted_edge_jaccard']:.3f}"
                if r.get("adjusted_edge_jaccard") is not None else "N/A"
            )
            raw_edge_jaccard_str = (
                f"{r['raw_edge_jaccard']:.3f}"
                if r.get("raw_edge_jaccard") is not None else "N/A"
            )
            score_str = (
                f"{r['final_score']:.3f}"
                if r.get("final_score") is not None else "N/A"
            )

            lines.append(
                f"| {r['dataset']} | {r['method']} | {runtime_str} | "
                f"{memory_str} | {r['node_detection_rate']:.3f} | "
                f"{r['edge_precision']:.3f} | {r['edge_recall']:.3f} | "
                f"{raw_edge_jaccard_str} | {aej_str} | "
                f"{r['division_jaccard']:.3f} | {score_str} |\n"
            )

        return "".join(lines)

def plot_benchmark_dashboard(rows):
    """
    One page, several small bar charts comparing methods side by side — Runtime, Memory,
    Precision/Recall, and Tracking Score — matching the "one page" dashboard shape rather than
    a single number standing alone. Returns the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [r["method"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    runtimes = [r["runtime_seconds"] if r.get("runtime_seconds") is not None else 0 for r in rows]
    runtime_known = [r.get("runtime_seconds") is not None for r in rows]
    axes[0, 0].bar(methods, runtimes,
                    color=["tab:blue" if k else "lightgray" for k in runtime_known])
    axes[0, 0].set_title("Runtime (s)" + ("" if all(runtime_known) else "  (gray = not measured)"))
    axes[0, 0].tick_params(axis="x", rotation=20)

    memory = [r["memory_delta_mb"] if r.get("memory_delta_mb") is not None else 0 for r in rows]
    memory_known = [r.get("memory_delta_mb") is not None for r in rows]
    axes[0, 1].bar(methods, memory,
                    color=["tab:purple" if k else "lightgray" for k in memory_known])
    axes[0, 1].set_title("Memory delta (MB)" + ("" if all(memory_known) else "  (gray = not measured)"))
    axes[0, 1].tick_params(axis="x", rotation=20)

    width = 0.35
    x = np.arange(len(methods))
    axes[1, 0].bar(x - width / 2, [r["edge_precision"] for r in rows], width, label="Precision")
    axes[1, 0].bar(x + width / 2, [r["edge_recall"] for r in rows], width, label="Recall")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(methods, rotation=20)
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_title("Edge Precision / Recall")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].bar(methods, [r["final_score"] for r in rows], color="tab:green")
    axes[1, 1].set_title("Tracking Score (adjusted edge Jaccard + 0.1 * division Jaccard)")
    axes[1, 1].tick_params(axis="x", rotation=20)

    fig.suptitle(f"Benchmark: {rows[0]['dataset']}" if rows else "Benchmark")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# TrackMate import — see the module docstring's caveat before trusting this.
# --------------------------------------------------------------------------

def load_trackmate_xml(path):
    """
    Parses a TrackMate XML model export (Model > AllSpots > SpotsInFrame > Spot, and
    Model > AllTracks > Track > Edge) into the (nodes, edges) shape src/evaluate.py expects.

    Written from TrackMate's documented XML schema. NOT tested against a real exported file —
    there's no Fiji/TrackMate available in the environment that wrote this. Before trusting a
    comparison table that includes a TrackMate row, export one real file from Fiji, run it
    through this function, and sanity-check the node/edge counts against what TrackMate's own
    UI reports for that file.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    nodes = []
    for spots_in_frame in root.iter("SpotsInFrame"):
        frame = int(spots_in_frame.get("frame"))
        for spot in spots_in_frame.iter("Spot"):
            nodes.append({
                # TrackMate's Spot ID is numeric (just serialized as an XML attribute string).
                # Cast to int here, once, rather than leaving it a string — src/evaluate.py's
                # node matching treats node_id as an opaque dict key so a string would technically
                # still "work" in isolation, but every other node source in this codebase
                # (LineageResult.to_dataframe(), run_real_benchmark.py's ground truth) uses int
                # IDs, and a silent type mismatch between "our" node_ids and TrackMate's is
                # exactly the kind of thing that only surfaces once you diff two real result sets.
                "node_id": int(spot.get("ID")),
                "t": frame,
                "y": float(spot.get("POSITION_Y")),
                "x": float(spot.get("POSITION_X")),
            })

    edges = []
    for track in root.iter("Track"):
        for edge in track.iter("Edge"):
            source = int(edge.get("SPOT_SOURCE_ID"))
            target = int(edge.get("SPOT_TARGET_ID"))
            edges.append((source, target))

    return nodes, edges