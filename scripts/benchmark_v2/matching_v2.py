"""
Distance-gated Hungarian matching for Benchmark v2.

Fixes the specific limitation identified in scripts/labos_diagnostic_benchmark.py's
`match_frame`: that function runs unconstrained Hungarian assignment (every GT point paired
with *some* candidate whenever one exists), so a true miss never reduces the "coverage" metric
-- it only shows up as one large distance buried in the mean/tail. Benchmark v2 makes misses
and false detections first-class, countable outcomes (FN / FP) instead of hidden inside a
distance average.

METHOD (capped-cost Hungarian, the standard way to do gated 1:1 assignment):
  1. D = cdist(gt_pts, pred_pts)                    -- true Euclidean distances
  2. cost = D, but any entry > d_max is replaced with a large penalty (LARGE_PENALTY)
  3. linear_sum_assignment(cost)                      -- solve on the CAPPED matrix
  4. For each solved pair, check the ORIGINAL distance (not the capped cost):
       - distance <= d_max  -> True Positive (TP), real match
       - distance >  d_max  -> not a real match; this GT point and this candidate are each
                                 treated as unmatched (see step 5)

Why cap instead of just running plain Hungarian on raw distances and thresholding afterward?
Plain (uncapped) Hungarian minimizes TOTAL summed distance across all pairs -- it can be
forced into a globally "cheaper" solution that pairs a GT point with a moderately-far
candidate instead of leaving both point sets to be counted separately, if doing so lowers the
sum elsewhere. Capping distances beyond d_max to a large, roughly-uniform penalty removes that
incentive: the solver still prefers any valid (<=d_max) pairing over an invalid one, but two
invalid pairings cost the solver about the same regardless of exactly how far apart they are,
so it no longer "trades off" real match quality against far-away matches it shouldn't be
making anyway. This is the standard approach used in multi-object-tracking evaluation
(e.g. CLEAR-MOT style gating).

  5. TP = # pairs with distance <= d_max
     FN = n_gt  - TP   (every GT point is either a TP or not; "not assigned at all" because
                         there were fewer candidates than GT points, and "assigned but too far",
                         both land here -- there is no double counting, see note below)
     FP = n_pred - TP  (symmetric logic for candidates)

Note on FN/FP arithmetic: because linear_sum_assignment always returns exactly
min(n_gt, n_pred) pairs, some GT or candidate points may never appear in the solved pairing at
all (whichever set is larger has leftover, unpaired points). Those leftovers are, by
definition, not TP -- so `FN = n_gt - TP` and `FP = n_pred - TP` correctly count them without
needing to enumerate "assigned-but-invalid" and "never-assigned" as separate cases.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

LARGE_PENALTY = 1e6


def gated_match(gt_pts, pred_pts, d_max):
    """
    Returns a dict:
        tp_distances : np.ndarray of the true (uncapped) distances for TP pairs only
        tp, fp, fn   : ints
    Handles empty gt_pts / pred_pts (returns all-FN or all-FP as appropriate, tp_distances
    empty) without calling the solver on a degenerate 0-sized matrix.
    """
    n_gt, n_pred = len(gt_pts), len(pred_pts)

    if n_gt == 0 and n_pred == 0:
        return {"tp_distances": np.array([]), "tp": 0, "fp": 0, "fn": 0}
    if n_gt == 0:
        return {"tp_distances": np.array([]), "tp": 0, "fp": n_pred, "fn": 0}
    if n_pred == 0:
        return {"tp_distances": np.array([]), "tp": 0, "fp": 0, "fn": n_gt}

    D = cdist(gt_pts, pred_pts)
    cost = np.where(D > d_max, LARGE_PENALTY, D)
    rows, cols = linear_sum_assignment(cost)

    matched_distances = D[rows, cols]
    valid = matched_distances <= d_max
    tp_distances = matched_distances[valid]

    tp = int(valid.sum())
    fn = n_gt - tp
    fp = n_pred - tp

    return {"tp_distances": tp_distances, "tp": tp, "fp": fp, "fn": fn}


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 and not (np.isnan(precision) or np.isnan(recall))
          else float("nan"))
    return precision, recall, f1
