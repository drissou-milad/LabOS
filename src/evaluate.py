"""
Implements the Biohub competition's actual scoring metric, transcribed directly from
https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md (fetched and
read in full — not a paraphrase from memory) so that "our method achieves X" can be a real,
checkable number, not a number that only sounds right.

Two honesty notes, on purpose:

1. Edge Jaccard / adjusted edge Jaccard (below) closely follow the published spec and are
   the higher-confidence, higher-weight (coefficient 1.0) part of the score. They have solid
   test coverage.

2. Division Jaccard's spec describes a "division subgraph" — "a GT node that splits into two
   children, with its parent and grandchildren included for context" — precisely enough to
   implement, but loosely enough that there's real interpretive judgment involved (exactly how
   far the subgraph extends). The implementation here reads it as: the dividing node's own
   parent (if any), the divider itself, its two children, and each child's own children
   (grandchildren of the divider) — i.e. one generation back, two generations forward. This is
   a best-effort reading, not a copy of an official reference implementation (none was found
   publicly), and it only carries 0.1 weight in the final score. Treat it with more scrutiny
   than the edge Jaccard before quoting it anywhere that matters.

Also assumed, not explicitly stated in the published spec but a near-certain requirement for
node matching to make sense at all: nodes are only matched to other nodes at the *same
timepoint* (a cell detected at t=3 can't be "the same cell" as one at t=5 by centroid distance
alone).
"""

from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------
# Node matching
# --------------------------------------------------------------------------

def match_nodes(pred_nodes, gt_nodes, max_distance=7.0):
    """
    pred_nodes, gt_nodes: lists of dicts with keys "node_id", "t", "y", "x" (matches
    LineageResult.to_dataframe()'s columns).

    Matches nodes by centroid distance within the same timepoint, via optimal bipartite
    assignment per timepoint, capped at max_distance. Each predicted node matches at most one
    ground-truth node (and vice versa).

    Returns dict: pred_node_id -> gt_node_id, for matched pairs only.
    """
    pred_by_t = defaultdict(list)
    gt_by_t = defaultdict(list)
    for n in pred_nodes:
        pred_by_t[n["t"]].append(n)
    for n in gt_nodes:
        gt_by_t[n["t"]].append(n)

    matches = {}
    for t in pred_by_t:
        if t not in gt_by_t:
            continue
        p_list = pred_by_t[t]
        g_list = gt_by_t[t]
        p_coords = np.array([[n["y"], n["x"]] for n in p_list])
        g_coords = np.array([[n["y"], n["x"]] for n in g_list])

        dist_matrix = cdist(p_coords, g_coords)
        rows, cols = linear_sum_assignment(dist_matrix)
        for r, c in zip(rows, cols):
            if dist_matrix[r, c] <= max_distance:
                matches[p_list[r]["node_id"]] = g_list[c]["node_id"]

    return matches


# --------------------------------------------------------------------------
# Edge Jaccard
# --------------------------------------------------------------------------

def edge_confusion(pred_edges, gt_edges, node_match, pred_nodes=None, gt_nodes=None):
    """
    Compute edge TP/FP/FN using the CTC edge-matching rules.

    Only consecutive-frame predicted edges are evaluated.
    Predicted edges that collapse onto the same matched GT edge are
    counted only once.

    If pred_nodes / gt_nodes are supplied, frame information is used to
    enforce the consecutive-frame rule. For backward compatibility,
    callers that do not supply node dictionaries fall back to evaluating
    all edges.
    """
    # Build node -> time lookup when node records are available.
    pred_time = {}
    if pred_nodes is not None:
        pred_time = {
            n["node_id"]: n["t"]
            for n in pred_nodes
        }

    gt_time = {}
    if gt_nodes is not None:
        gt_time = {
            n["node_id"]: n["t"]
            for n in gt_nodes
        }

    # Official metric evaluates only consecutive-frame GT edges.
    valid_gt_edges = set()
    for s, t in gt_edges:
        if gt_time:
            if gt_time.get(t) != gt_time.get(s, -10**9) + 1:
                continue
        valid_gt_edges.add((s, t))

    gt_targets_of_source = defaultdict(set)
    gt_sources_of_target = defaultdict(set)

    for s, t in valid_gt_edges:
        gt_targets_of_source[s].add(t)
        gt_sources_of_target[t].add(s)

    tp = 0
    fp = 0
    matched_gt_edges = set()
    evaluated_pred_edges = set()

    for ps, pt in pred_edges:
        # Official metric only evaluates consecutive-frame predicted edges.
        if pred_time:
            if pred_time.get(pt) != pred_time.get(ps, -10**9) + 1:
                continue

        # Ignore exact duplicate predicted edges.
        pred_edge = (ps, pt)
        if pred_edge in evaluated_pred_edges:
            continue
        evaluated_pred_edges.add(pred_edge)

        gs = node_match.get(ps)
        gtid = node_match.get(pt)

        # True positive.
        if (
            gs is not None
            and gtid is not None
            and gtid in gt_targets_of_source.get(gs, set())
        ):
            matched_gt_edge = (gs, gtid)

            # Do not count multiple predictions mapping to the same GT edge.
            if matched_gt_edge not in matched_gt_edges:
                tp += 1
                matched_gt_edges.add(matched_gt_edge)

            continue

        # FP condition (a):
        # target matches a GT node connected to another source.
        target_is_fp = (
            gtid is not None
            and len(gt_sources_of_target.get(gtid, ())) > 0
            and (
                gs is None
                or gtid not in gt_targets_of_source.get(gs, set())
            )
        )

        # FP condition (b):
        # source matches a GT node connected to another target.
        source_is_fp = (
            gs is not None
            and len(gt_targets_of_source.get(gs, ())) > 0
            and (
                gtid is None
                or gtid not in gt_targets_of_source.get(gs, set())
            )
        )

        if target_is_fp or source_is_fp:
            fp += 1

    # Every valid GT edge not recovered by a prediction is an FN.
    fn = len(valid_gt_edges) - len(matched_gt_edges)

    return tp, fp, fn


def edge_jaccard(tp, fp, fn):
    """
    Compute edge Jaccard from TP, FP and FN.
    """
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0

def adjusted_edge_jaccard(jaccard, n_pred_nodes, n_true_nodes_estimate, a=0.1):
    """
    n_true_nodes_estimate: a *coarse estimate* of the total true node count (including
    ground truth doesn't annotate) — this has to come from outside this function (the
    competition provides it per-sample); there's no way to derive it from sparse ground truth
    alone.
    """
    if n_true_nodes_estimate <= 0:
        return jaccard
    penalty = 1 - a * (n_pred_nodes - n_true_nodes_estimate) / n_true_nodes_estimate
    return max(0.0, jaccard * penalty)


def aggregate_adjusted_edge_jaccard(per_sample):
    """
    per_sample: list of dicts with keys tp, fp, fn, n_pred_nodes, n_true_nodes_estimate.
    Returns the sample-size-weighted average adjusted edge Jaccard (w_i = tp+fp+fn), matching
    the spec's micro-averaging.
    """
    total_weighted = 0.0
    total_weight = 0
    for s in per_sample:
        w = s["tp"] + s["fp"] + s["fn"]
        j = edge_jaccard(s["tp"], s["fp"], s["fn"])
        aj = adjusted_edge_jaccard(j, s["n_pred_nodes"], s["n_true_nodes_estimate"])
        total_weighted += w * aj
        total_weight += w
    return total_weighted / total_weight if total_weight > 0 else 0.0


# --------------------------------------------------------------------------
# Division Jaccard — see the module-level caveat about this section.
# --------------------------------------------------------------------------

def _division_subgraph_nodes(divider_gt_id, gt_edges_by_source, gt_edges_by_target):
    """One generation back, two generations forward from the dividing GT node — see the
    module docstring for why this specific span."""
    nodes = {divider_gt_id}
    parents = gt_edges_by_target.get(divider_gt_id, set())
    nodes |= parents

    children = gt_edges_by_source.get(divider_gt_id, set())
    nodes |= children
    for c in children:
        nodes |= gt_edges_by_source.get(c, set())  # grandchildren

    return nodes


def find_divisions(edges):
    """Nodes with exactly two outgoing edges, per the spec's definition of a division."""
    children_of = defaultdict(set)
    for s, t in edges:
        children_of[s].add(t)
    return {n: tuple(sorted(c)) for n, c in children_of.items() if len(c) == 2}


def division_confusion(pred_edges, gt_edges, node_match):
    """
    node_match: dict pred_node_id -> gt_node_id, from match_nodes().
    Returns (tp, fp, fn) for division events, per the spec's division-matching rule
    (see the module docstring's caveat on this function's interpretive judgment calls).
    """
    gt_edges_by_source = defaultdict(set)
    gt_edges_by_target = defaultdict(set)
    for s, t in gt_edges:
        gt_edges_by_source[s].add(t)
        gt_edges_by_target[t].add(s)

    gt_divisions = find_divisions(gt_edges)
    pred_divisions = find_divisions(pred_edges)

    # For connected-component checks on the *predicted* graph.
    pred_neighbors = defaultdict(set)
    for s, t in pred_edges:
        pred_neighbors[s].add(t)
        pred_neighbors[t].add(s)

    def predicted_component(start):
        seen = {start}
        stack = [start]
        while stack:
            n = stack.pop()
            for nb in pred_neighbors.get(n, ()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return seen

    gt_to_pred = defaultdict(set)
    for p, g in node_match.items():
        gt_to_pred[g].add(p)

    tp = 0
    fn = 0
    matched_gt_divisions = set()

    for divider, (c1, c2) in gt_divisions.items():
        subgraph = _division_subgraph_nodes(divider, gt_edges_by_source, gt_edges_by_target)
        lineage1 = {c1} | gt_edges_by_source.get(c1, set())
        lineage2 = {c2} | gt_edges_by_source.get(c2, set())

        # Any predicted node matched to something in the pre-split part of the subgraph
        # (divider or its parent).
        pre_split_gt = {divider} | gt_edges_by_target.get(divider, set())
        pre_split_pred_nodes = set()
        for g in pre_split_gt:
            pre_split_pred_nodes |= gt_to_pred.get(g, set())

        lineage1_pred_nodes = set()
        for g in lineage1 & subgraph:
            lineage1_pred_nodes |= gt_to_pred.get(g, set())
        lineage2_pred_nodes = set()
        for g in lineage2 & subgraph:
            lineage2_pred_nodes |= gt_to_pred.get(g, set())

        found = False
        if pre_split_pred_nodes and lineage1_pred_nodes and lineage2_pred_nodes:
            # All matched nodes so far, in one predicted connected component, containing a
            # predicted fork.
            all_matched = pre_split_pred_nodes | lineage1_pred_nodes | lineage2_pred_nodes
            components_checked = set()
            for start_node in all_matched:
                if start_node in components_checked:
                    continue
                component = predicted_component(start_node)
                components_checked |= component
                if all_matched <= component:
                    has_fork = any(n in pred_divisions for n in component)
                    if has_fork:
                        found = True
                        break

        if found:
            tp += 1
            matched_gt_divisions.add(divider)
        else:
            fn += 1

    # FP: a predicted division whose matched region lands in an annotated area but wasn't
    # paired to any GT division above.
    fp = 0
    for pred_divider in pred_divisions:
        gt_id = node_match.get(pred_divider)
        if gt_id is None:
            continue  # unannotated region — ignored, per the edge rule this mirrors
        # "annotated area": the matched GT node participates in at least one GT edge
        is_annotated = gt_id in gt_edges_by_source or gt_id in gt_edges_by_target
        if not is_annotated:
            continue
        # Was this predicted division used to satisfy any matched_gt_divisions above?
        component = predicted_component(pred_divider)
        satisfies_some_match = False
        for divider in matched_gt_divisions:
            pre_split_gt = {divider} | gt_edges_by_target.get(divider, set())
            pre_split_pred_nodes = set()
            for g in pre_split_gt:
                pre_split_pred_nodes |= gt_to_pred.get(g, set())
            if pre_split_pred_nodes & component or component & {
                p for g in gt_divisions.get(divider, ()) for p in gt_to_pred.get(g, set())
            }:
                satisfies_some_match = True
                break
        if not satisfies_some_match:
            fp += 1

    return tp, fp, fn


def division_jaccard(tp, fp, fn):
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0


# --------------------------------------------------------------------------
# Final combined score
# --------------------------------------------------------------------------

def final_score(adjusted_edge_jaccard_value, division_jaccard_value, w=0.1):
    return adjusted_edge_jaccard_value + w * division_jaccard_value