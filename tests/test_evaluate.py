from src.evaluate import (
    match_nodes,
    edge_confusion,
    edge_jaccard,
    adjusted_edge_jaccard,
    aggregate_adjusted_edge_jaccard,
    division_confusion,
    division_jaccard,
    final_score,
)
from src.tracker import HungarianTracker
from src.lineage import LineageBuilder


def _gt_chain():
    nodes = [
        {"node_id": "g0", "t": 0, "y": 0.0, "x": 0.0},
        {"node_id": "g1", "t": 1, "y": 1.0, "x": 1.0},
        {"node_id": "g2", "t": 2, "y": 2.0, "x": 2.0},
    ]
    edges = [("g0", "g1"), ("g1", "g2")]
    return nodes, edges


def test_node_matching_respects_max_distance():
    gt_nodes = [{"node_id": "g0", "t": 0, "y": 0.0, "x": 0.0}]
    pred_nodes = [{"node_id": "p0", "t": 0, "y": 10.0, "x": 10.0}]  # far away
    matches = match_nodes(pred_nodes, gt_nodes, max_distance=1.0)
    assert matches == {}


def test_node_matching_only_within_same_timepoint():
    gt_nodes = [{"node_id": "g0", "t": 0, "y": 0.0, "x": 0.0}]
    pred_nodes = [{"node_id": "p0", "t": 1, "y": 0.0, "x": 0.0}]  # same place, different t
    matches = match_nodes(pred_nodes, gt_nodes, max_distance=100.0)
    assert matches == {}


def test_perfect_prediction_gives_jaccard_one():
    gt_nodes, gt_edges = _gt_chain()
    pred_nodes = [
        {"node_id": "p0", "t": 0, "y": 0.1, "x": 0.1},
        {"node_id": "p1", "t": 1, "y": 1.1, "x": 0.9},
        {"node_id": "p2", "t": 2, "y": 2.1, "x": 1.9},
    ]
    pred_edges = [("p0", "p1"), ("p1", "p2")]

    nm = match_nodes(pred_nodes, gt_nodes, max_distance=1.0)
    tp, fp, fn = edge_confusion(pred_edges, gt_edges, nm)
    assert (tp, fp, fn) == (2, 0, 0)
    assert edge_jaccard(tp, fp, fn) == 1.0


def test_missed_edge_is_false_negative():
    gt_nodes, gt_edges = _gt_chain()
    pred_nodes = [
        {"node_id": "p0", "t": 0, "y": 0.1, "x": 0.1},
        {"node_id": "p1", "t": 1, "y": 1.1, "x": 0.9},
        {"node_id": "p2", "t": 2, "y": 2.1, "x": 1.9},
    ]
    pred_edges = [("p0", "p1")]  # missing the second edge entirely

    nm = match_nodes(pred_nodes, gt_nodes, max_distance=1.0)
    tp, fp, fn = edge_confusion(pred_edges, gt_edges, nm)
    assert (tp, fp, fn) == (1, 0, 1)


def test_wrong_link_is_false_positive_and_false_negative():
    """Predicting an edge to the wrong (but matched) target should count as both an FP
    (wrong edge) and an FN (the real edge was never predicted)."""
    gt_nodes = [
        {"node_id": "g0", "t": 0, "y": 0.0, "x": 0.0},
        {"node_id": "g1", "t": 1, "y": 1.0, "x": 1.0},
        {"node_id": "g2", "t": 1, "y": 50.0, "x": 50.0},
    ]
    gt_edges = [("g0", "g1")]  # g2 is unrelated, no edge from g0

    pred_nodes = [
        {"node_id": "p0", "t": 0, "y": 0.1, "x": 0.1},
        {"node_id": "p2", "t": 1, "y": 50.1, "x": 50.1},
    ]
    pred_edges = [("p0", "p2")]  # wrongly linked to the unrelated node

    nm = match_nodes(pred_nodes, gt_nodes, max_distance=1.0)
    tp, fp, fn = edge_confusion(pred_edges, gt_edges, nm)
    assert (tp, fp, fn) == (0, 1, 1)


def test_unmatched_predicted_nodes_are_not_penalized():
    """Sparse ground truth: predicted nodes/edges with no GT counterpart at all are ignored,
    not counted as false positives (per the spec)."""
    gt_nodes, gt_edges = _gt_chain()
    pred_nodes = [
        {"node_id": "p0", "t": 0, "y": 0.1, "x": 0.1},
        {"node_id": "p1", "t": 1, "y": 1.1, "x": 0.9},
        {"node_id": "p2", "t": 2, "y": 2.1, "x": 1.9},
        {"node_id": "pX", "t": 0, "y": 500.0, "x": 500.0},  # no GT counterpart anywhere nearby
        {"node_id": "pY", "t": 1, "y": 500.0, "x": 500.0},
    ]
    pred_edges = [("p0", "p1"), ("p1", "p2"), ("pX", "pY")]  # extra, unrelated edge

    nm = match_nodes(pred_nodes, gt_nodes, max_distance=1.0)
    tp, fp, fn = edge_confusion(pred_edges, gt_edges, nm)
    assert (tp, fp, fn) == (2, 0, 0)  # the extra edge changes nothing


def test_adjusted_jaccard_penalizes_overprediction():
    perfect = adjusted_edge_jaccard(1.0, n_pred_nodes=10, n_true_nodes_estimate=10)
    over = adjusted_edge_jaccard(1.0, n_pred_nodes=20, n_true_nodes_estimate=10)
    assert perfect == 1.0
    assert abs(over - 0.9) < 1e-9  # 100% over-prediction, a=0.1 -> penalty factor 0.9


def test_adjusted_jaccard_never_goes_negative():
    result = adjusted_edge_jaccard(1.0, n_pred_nodes=1000, n_true_nodes_estimate=10)
    assert result >= 0.0


def test_aggregate_weights_by_sample_size():
    """A sample with more TP+FP+FN should count more in the weighted average than a tiny one."""
    per_sample = [
        {"tp": 100, "fp": 0, "fn": 0, "n_pred_nodes": 10, "n_true_nodes_estimate": 10},  # jaccard 1.0, big
        {"tp": 0, "fp": 1, "fn": 0, "n_pred_nodes": 10, "n_true_nodes_estimate": 10},     # jaccard 0.0, tiny
    ]
    result = aggregate_adjusted_edge_jaccard(per_sample)
    assert result > 0.9  # dominated by the large, perfect sample


def test_division_perfect_prediction():
    gt_nodes = [
        {"node_id": "gA", "t": 0, "y": 0.0, "x": 0.0},
        {"node_id": "gB", "t": 1, "y": 5.0, "x": 5.0},
        {"node_id": "gC", "t": 1, "y": -5.0, "x": -5.0},
        {"node_id": "gD", "t": 2, "y": 6.0, "x": 6.0},
        {"node_id": "gE", "t": 2, "y": -6.0, "x": -6.0},
    ]
    gt_edges = [("gA", "gB"), ("gA", "gC"), ("gB", "gD"), ("gC", "gE")]
    pred_nodes = [
        {"node_id": "pA", "t": 0, "y": 0.1, "x": 0.1},
        {"node_id": "pB", "t": 1, "y": 5.1, "x": 5.1},
        {"node_id": "pC", "t": 1, "y": -5.1, "x": -5.1},
        {"node_id": "pD", "t": 2, "y": 6.1, "x": 6.1},
        {"node_id": "pE", "t": 2, "y": -6.1, "x": -6.1},
    ]
    pred_edges = [("pA", "pB"), ("pA", "pC"), ("pB", "pD"), ("pC", "pE")]

    nm = match_nodes(pred_nodes, gt_nodes, max_distance=1.0)
    tp, fp, fn = division_confusion(pred_edges, gt_edges, nm)
    assert (tp, fp, fn) == (1, 0, 0)
    assert division_jaccard(tp, fp, fn) == 1.0


def test_division_missed_is_false_negative():
    gt_nodes = [
        {"node_id": "gA", "t": 0, "y": 0.0, "x": 0.0},
        {"node_id": "gB", "t": 1, "y": 5.0, "x": 5.0},
        {"node_id": "gC", "t": 1, "y": -5.0, "x": -5.0},
        {"node_id": "gD", "t": 2, "y": 6.0, "x": 6.0},
        {"node_id": "gE", "t": 2, "y": -6.0, "x": -6.0},
    ]
    gt_edges = [("gA", "gB"), ("gA", "gC"), ("gB", "gD"), ("gC", "gE")]
    pred_nodes = [
        {"node_id": "pA", "t": 0, "y": 0.1, "x": 0.1},
        {"node_id": "pB", "t": 1, "y": 5.1, "x": 5.1},
        {"node_id": "pD", "t": 2, "y": 6.1, "x": 6.1},
    ]
    pred_edges = [("pA", "pB"), ("pB", "pD")]  # only one daughter tracked, no fork predicted

    nm = match_nodes(pred_nodes, gt_nodes, max_distance=1.0)
    tp, fp, fn = division_confusion(pred_edges, gt_edges, nm)
    assert (tp, fp, fn) == (0, 0, 1)


def test_division_false_positive_in_annotated_region():
    gt_nodes = [
        {"node_id": "gA", "t": 0, "y": 0.0, "x": 0.0},
        {"node_id": "gB", "t": 1, "y": 1.0, "x": 1.0},
        {"node_id": "gC", "t": 2, "y": 2.0, "x": 2.0},
    ]
    gt_edges = [("gA", "gB"), ("gB", "gC")]  # no division in ground truth

    pred_nodes = [
        {"node_id": "pA", "t": 0, "y": 0.1, "x": 0.1},
        {"node_id": "pB", "t": 1, "y": 1.1, "x": 1.1},
        {"node_id": "pX", "t": 1, "y": 40.0, "x": 40.0},  # spurious extra child
        {"node_id": "pC", "t": 2, "y": 2.1, "x": 2.1},
    ]
    pred_edges = [("pA", "pB"), ("pA", "pX"), ("pB", "pC")]  # invents a fork at pA

    nm = match_nodes(pred_nodes, gt_nodes, max_distance=1.0)
    tp, fp, fn = division_confusion(pred_edges, gt_edges, nm)
    assert (tp, fp, fn) == (0, 1, 0)


def test_final_score_combines_with_correct_weight():
    score = final_score(adjusted_edge_jaccard_value=0.8, division_jaccard_value=0.5, w=0.1)
    assert abs(score - 0.85) < 1e-9


def test_integration_with_lineage_builder():
    """The actual point: LineageBuilder's real output, evaluated against a hand-built ground
    truth, using the same dict/tuple shapes end to end — proving these two modules actually
    fit together, not just that each one works in isolation."""
    import numpy as np

    frames = [
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 1.0]]),
        np.array([[2.0, 2.0], [12.0, 12.0]]),  # division
    ]
    tracker = HungarianTracker(max_distance=5)
    builder = LineageBuilder(tracker, division_max_distance=20)
    result = builder.build(frames)

    pred_df = result.to_dataframe()
    pred_nodes = pred_df[["node_id", "t", "y", "x"]].to_dict("records")
    pred_edges = result.edges

    # Ground truth: matches the prediction exactly (a "perfect prediction" sanity check —
    # this isn't asserting the algorithm is right, just that the two modules interoperate).
    gt_nodes = [{"node_id": f"gt_{n['node_id']}", "t": n["t"], "y": n["y"], "x": n["x"]} for n in pred_nodes]
    id_map = {n["node_id"]: f"gt_{n['node_id']}" for n in pred_nodes}
    gt_edges = [(id_map[s], id_map[t]) for s, t in pred_edges]

    nm = match_nodes(pred_nodes, gt_nodes, max_distance=1.0)
    assert len(nm) == len(pred_nodes)

    tp, fp, fn = edge_confusion(pred_edges, gt_edges, nm)
    assert edge_jaccard(tp, fp, fn) == 1.0

    div_tp, div_fp, div_fn = division_confusion(pred_edges, gt_edges, nm)
    assert division_jaccard(div_tp, div_fp, div_fn) == 1.0

def test_nonconsecutive_predicted_edge_is_ignored():
    gt_nodes = [
        {"node_id": "g0", "t": 0, "y": 0.0, "x": 0.0},
        {"node_id": "g1", "t": 1, "y": 1.0, "x": 1.0},
        {"node_id": "g2", "t": 2, "y": 2.0, "x": 2.0},
    ]
    gt_edges = [("g0", "g1"), ("g1", "g2")]

    pred_nodes = [
        {"node_id": "p0", "t": 0, "y": 0.0, "x": 0.0},
        {"node_id": "p2", "t": 2, "y": 2.0, "x": 2.0},
    ]
    pred_edges = [("p0", "p2")]  # skips frame 1

    nm = match_nodes(pred_nodes, gt_nodes, max_distance=1.0)
    tp, fp, fn = edge_confusion(
        pred_edges,
        gt_edges,
        nm,
        pred_nodes=pred_nodes,
        gt_nodes=gt_nodes,
    )

    assert (tp, fp, fn) == (0, 0, 2)


def test_duplicate_predicted_edges_do_not_create_duplicate_tp():
    gt_nodes, gt_edges = _gt_chain()

    pred_nodes = [
        {"node_id": "p0", "t": 0, "y": 0.0, "x": 0.0},
        {"node_id": "p1", "t": 1, "y": 1.0, "x": 1.0},
        {"node_id": "p2", "t": 2, "y": 2.0, "x": 2.0},
    ]

    pred_edges = [
        ("p0", "p1"),
        ("p0", "p1"),  # duplicate prediction
        ("p1", "p2"),
    ]

    nm = match_nodes(pred_nodes, gt_nodes, max_distance=1.0)
    tp, fp, fn = edge_confusion(pred_edges, gt_edges, nm)

    assert (tp, fp, fn) == (2, 0, 0)