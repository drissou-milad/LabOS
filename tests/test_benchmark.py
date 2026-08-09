from pathlib import Path

import numpy as np

from src.tracker import HungarianTracker
from src.benchmark import (
    NaiveGreedyTracker, run_method, run_benchmark, format_results_table,
    score_external_method, plot_benchmark_dashboard, load_trackmate_xml, score_method,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _synthetic_dataset():
    frames = [
        np.array([[10.0, 10.0], [40.0, 40.0]]),
        np.array([[11.0, 11.0], [41.0, 41.0]]),
        np.array([[12.0, 12.0], [20.0, 20.0], [42.0, 42.0]]),  # division
        np.array([[13.0, 13.0], [21.0, 21.0], [43.0, 43.0]]),
    ]
    gt_nodes = [
        {"node_id": "g0_0", "t": 0, "y": 10.0, "x": 10.0},
        {"node_id": "g1_0", "t": 0, "y": 40.0, "x": 40.0},
        {"node_id": "g0_1", "t": 1, "y": 11.0, "x": 11.0},
        {"node_id": "g1_1", "t": 1, "y": 41.0, "x": 41.0},
        {"node_id": "g0_2a", "t": 2, "y": 12.0, "x": 12.0},
        {"node_id": "g0_2b", "t": 2, "y": 20.0, "x": 20.0},
        {"node_id": "g1_2", "t": 2, "y": 42.0, "x": 42.0},
        {"node_id": "g0_3a", "t": 3, "y": 13.0, "x": 13.0},
        {"node_id": "g0_3b", "t": 3, "y": 21.0, "x": 21.0},
        {"node_id": "g1_3", "t": 3, "y": 43.0, "x": 43.0},
    ]
    gt_edges = [
        ("g0_0", "g0_1"), ("g1_0", "g1_1"),
        ("g0_1", "g0_2a"), ("g0_1", "g0_2b"),
        ("g1_1", "g1_2"),
        ("g0_2a", "g0_3a"), ("g0_2b", "g0_3b"), ("g1_2", "g1_3"),
    ]
    return frames, gt_nodes, gt_edges


def test_naive_greedy_tracker_matches_within_range():
    tracker = NaiveGreedyTracker(max_distance=5)
    previous = np.array([[0.0, 0.0]])
    current = np.array([[1.0, 1.0]])
    matches = tracker.match(previous, current)
    assert len(matches) == 1
    assert matches[0]["previous"] == 0
    assert matches[0]["current"] == 0


def test_naive_greedy_tracker_respects_max_distance():
    tracker = NaiveGreedyTracker(max_distance=1)
    previous = np.array([[0.0, 0.0]])
    current = np.array([[50.0, 50.0]])
    assert tracker.match(previous, current) == []


def test_naive_greedy_tracker_does_not_double_assign():
    """Two previous points competing for the same nearby current point should not both grab
    it — this is exactly the kind of thing that distinguishes 'naive' from 'optimal'."""
    tracker = NaiveGreedyTracker(max_distance=10)
    previous = np.array([[0.0, 0.0], [0.5, 0.5]])
    current = np.array([[0.2, 0.2]])
    matches = tracker.match(previous, current)
    assert len(matches) == 1
    used_curr = {m["current"] for m in matches}
    assert len(used_curr) == len(matches)


def test_run_method_without_lineage_builder_produces_valid_nodes_and_edges():
    frames, gt_nodes, gt_edges = _synthetic_dataset()
    tracker = NaiveGreedyTracker(max_distance=5)
    nodes, edges, elapsed, _mem = run_method("baseline", frames, tracker, use_lineage_builder=False)

    assert len(nodes) > 0
    node_ids = {n["node_id"] for n in nodes}
    for s, t in edges:
        assert s in node_ids and t in node_ids
    assert elapsed >= 0


def test_run_method_with_lineage_builder_produces_valid_nodes_and_edges():
    frames, gt_nodes, gt_edges = _synthetic_dataset()
    tracker = HungarianTracker(max_distance=5)
    nodes, edges, elapsed, _mem = run_method(
        "ours", frames, tracker, use_lineage_builder=True, division_max_distance=15
    )

    assert len(nodes) > 0
    node_ids = {n["node_id"] for n in nodes}
    for s, t in edges:
        assert s in node_ids and t in node_ids
    assert elapsed >= 0


def test_run_benchmark_produces_one_row_per_method():
    frames, gt_nodes, gt_edges = _synthetic_dataset()
    methods = {
        "baseline": (NaiveGreedyTracker(max_distance=5), False, None),
        "ours": (HungarianTracker(max_distance=5), True, 15),
    }
    rows = run_benchmark("synthetic", frames, gt_nodes, gt_edges, methods,
                          n_true_nodes_estimate=len(gt_nodes), max_distance=2.0)
    assert len(rows) == 2
    methods_seen = {r["method"] for r in rows}
    assert methods_seen == {"baseline", "ours"}
    for r in rows:
        assert 0.0 <= r["division_jaccard"] <= 1.0
        assert r["runtime_seconds"] >= 0


def test_lineage_builder_catches_division_naive_baseline_does_not():
    """The actual point of building LineageBuilder in the first place — this should show up
    as a real, measurable difference, not just an assumption."""
    frames, gt_nodes, gt_edges = _synthetic_dataset()
    methods = {
        "baseline": (NaiveGreedyTracker(max_distance=5), False, None),
        "ours": (HungarianTracker(max_distance=5), True, 15),
    }
    rows = run_benchmark("synthetic", frames, gt_nodes, gt_edges, methods,
                          n_true_nodes_estimate=len(gt_nodes), max_distance=2.0)
    by_method = {r["method"]: r for r in rows}
    assert by_method["baseline"]["division_jaccard"] == 0.0
    assert by_method["ours"]["division_jaccard"] == 1.0


def test_run_method_returns_memory_delta():
    frames, gt_nodes, gt_edges = _synthetic_dataset()
    tracker = NaiveGreedyTracker(max_distance=5)
    nodes, edges, elapsed, memory_delta_mb = run_method("baseline", frames, tracker, use_lineage_builder=False)
    assert isinstance(memory_delta_mb, float)  # can be negative (GC reclaimed more than allocated) — just must be a real number


def test_run_benchmark_includes_precision_recall_and_memory():
    frames, gt_nodes, gt_edges = _synthetic_dataset()
    methods = {
        "baseline": (NaiveGreedyTracker(max_distance=5), False, None),
        "ours": (HungarianTracker(max_distance=5), True, 15),
    }
    rows = run_benchmark("synthetic", frames, gt_nodes, gt_edges, methods,
                          n_true_nodes_estimate=len(gt_nodes), max_distance=2.0)
    for r in rows:
        assert 0.0 <= r["edge_precision"] <= 1.0
        assert 0.0 <= r["edge_recall"] <= 1.0
        assert 0.0 <= r["node_detection_rate"] <= 1.0
        assert "memory_delta_mb" in r


def test_perfect_predictions_have_precision_and_recall_of_one():
    """Our method on this synthetic dataset should predict every ground-truth edge exactly —
    precision and recall should both be 1.0, not just the combined score."""
    frames, gt_nodes, gt_edges = _synthetic_dataset()
    methods = {"ours": (HungarianTracker(max_distance=5), True, 15)}
    rows = run_benchmark("synthetic", frames, gt_nodes, gt_edges, methods,
                          n_true_nodes_estimate=len(gt_nodes), max_distance=2.0)
    assert rows[0]["edge_precision"] == 1.0
    assert rows[0]["edge_recall"] == 1.0


def test_score_external_method_without_timing_reports_none():
    """An externally-computed result (e.g. from load_trackmate_xml) with no runtime/memory
    numbers should report them as None, not a fabricated 0."""
    frames, gt_nodes, gt_edges = _synthetic_dataset()
    row = score_external_method(
        "synthetic", "external tool", gt_nodes, gt_edges, gt_nodes, gt_edges,
        n_true_nodes_estimate=len(gt_nodes), max_distance=2.0,
    )
    assert row["runtime_seconds"] is None
    assert row["memory_delta_mb"] is None
    assert row["edge_precision"] == 1.0  # perfect self-match sanity check


def test_score_external_method_with_provided_timing():
    frames, gt_nodes, gt_edges = _synthetic_dataset()
    row = score_external_method(
        "synthetic", "external tool", gt_nodes, gt_edges, gt_nodes, gt_edges,
        n_true_nodes_estimate=len(gt_nodes), max_distance=2.0,
        runtime_seconds=12.5, memory_delta_mb=340.0,
    )
    assert row["runtime_seconds"] == 12.5
    assert row["memory_delta_mb"] == 340.0


def test_format_results_table_handles_missing_runtime_and_memory():
    rows = [{
        "dataset": "d1", "method": "external", "final_score": 0.5,
        "adjusted_edge_jaccard": 0.4, "division_jaccard": 0.1,
        "edge_precision": 0.9, "edge_recall": 0.8, "node_detection_rate": 0.95,
        "runtime_seconds": None, "memory_delta_mb": None,
    }]
    table = format_results_table(rows)
    assert "n/a" in table


def test_plot_benchmark_dashboard_does_not_crash():
    frames, gt_nodes, gt_edges = _synthetic_dataset()
    methods = {
        "baseline": (NaiveGreedyTracker(max_distance=5), False, None),
        "ours": (HungarianTracker(max_distance=5), True, 15),
    }
    rows = run_benchmark("synthetic", frames, gt_nodes, gt_edges, methods,
                          n_true_nodes_estimate=len(gt_nodes), max_distance=2.0)
    fig = plot_benchmark_dashboard(rows)
    assert fig is not None
    assert len(fig.axes) == 4


def test_plot_benchmark_dashboard_handles_missing_runtime_gracefully():
    frames, gt_nodes, gt_edges = _synthetic_dataset()
    row = score_external_method(
        "synthetic", "external", gt_nodes, gt_edges, gt_nodes, gt_edges,
        n_true_nodes_estimate=len(gt_nodes), max_distance=2.0,
    )
    fig = plot_benchmark_dashboard([row])  # no crash despite runtime_seconds=None
    assert fig is not None


def test_format_results_table_is_valid_markdown_table():
    rows = [{
        "dataset": "d1", "method": "m1", "final_score": 0.5,
        "adjusted_edge_jaccard": 0.4, "division_jaccard": 0.1, "runtime_seconds": 1.234,
        "memory_delta_mb": 5.0, "edge_precision": 0.9, "edge_recall": 0.8, "node_detection_rate": 0.95,
    }]
    table = format_results_table(rows)
    assert table.startswith("| Dataset |")
    assert "m1" in table
    assert "0.500" in table
    assert "1.234s" in table


# --------------------------------------------------------------------------
# load_trackmate_xml
#
# IMPORTANT: tests/fixtures/trackmate_sample.xml is hand-authored from TrackMate's documented
# XML schema, not exported from a real Fiji/TrackMate run (none is available in this
# environment). These tests prove load_trackmate_xml() correctly implements the schema it
# claims to -- they do NOT prove it survives whatever a real TrackMate export actually looks
# like in practice (extra attributes, a different TrackMate version's schema quirks, etc.).
# Re-run this kind of check against a real exported file before trusting a comparison table
# that includes a real TrackMate row -- see RUNBOOK.md step 8.
# --------------------------------------------------------------------------

def test_load_trackmate_xml_parses_all_spots_as_nodes():
    nodes, edges = load_trackmate_xml(FIXTURES_DIR / "trackmate_sample.xml")
    assert len(nodes) == 10
    # Spot ID=0 is at frame 0, (y=10.0, x=10.0) in the fixture.
    spot0 = next(n for n in nodes if n["node_id"] == 0)
    assert spot0 == {"node_id": 0, "t": 0, "y": 10.0, "x": 10.0}


def test_load_trackmate_xml_node_ids_are_ints_not_strings():
    """A silent str/int node_id mismatch against this project's own node sources (which are
    all int) would make every edge look unmatched without raising anything -- verify the cast
    happens, not just that parsing runs."""
    nodes, edges = load_trackmate_xml(FIXTURES_DIR / "trackmate_sample.xml")
    assert all(isinstance(n["node_id"], int) for n in nodes)
    assert all(isinstance(s, int) and isinstance(t, int) for s, t in edges)


def test_load_trackmate_xml_parses_edges_from_both_tracks():
    nodes, edges = load_trackmate_xml(FIXTURES_DIR / "trackmate_sample.xml")
    assert len(edges) == 8  # 5 from Track_0 (incl. the division) + 3 from Track_1
    assert (0, 2) in edges  # Track_0's first link
    assert (2, 4) in edges and (2, 5) in edges  # the division: spot 2 -> spots 4 and 5
    assert (1, 3) in edges  # Track_1's first link


def test_load_trackmate_xml_result_scores_perfectly_against_itself():
    """Sanity check in the same spirit as test_score_external_method_without_timing_reports_none:
    scoring the fixture's own parsed output against itself as "ground truth" should be a
    perfect match -- proves the parsed (nodes, edges) shape is actually usable by
    src/evaluate.py's scoring, not just structurally present."""
    nodes, edges = load_trackmate_xml(FIXTURES_DIR / "trackmate_sample.xml")
    scores = score_method(nodes, edges, nodes, edges, n_true_nodes_estimate=len(nodes), max_distance=2.0)
    assert scores["edge_precision"] == 1.0
    assert scores["edge_recall"] == 1.0
    assert scores["division_jaccard"] == 1.0
