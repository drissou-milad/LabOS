"""
Experiment #5: matched-contrast diagnostic. THE core Run08-vs-Run09 comparison tool.

Runs the EXACT SAME methodology against any run directory containing a
test_predictions_resumable.csv (Run08, Run09, or any future run) -- this is what makes the
Run08 vs Run09 comparison numbers actually comparable, per the explicit requirement to "use
exactly the same evaluation methodology for Run08 and Run09." There is exactly ONE
implementation of the matched-contrast algorithm in this codebase; it is never duplicated or
reimplemented per-run.

METHODOLOGY (identical for every run this script is pointed at):
    For every POSITIVE test patch, find the K=20 NEGATIVE test patches (pooled across the whole
    test set, not restricted to the same biological sample) whose raw std (pre-normalization,
    computed straight from the .npy pixel data) is closest to this positive's raw std. Report:
        - mean / median positive probability
        - mean / median matched-negative probability (each positive's own 20-neighbor mean/median,
          then averaged across positives)
        - % of positives whose probability exceeds their own matched-negative-group MEAN
        - % of positives whose probability exceeds their own matched-negative-group MEDIAN
        - the full distribution of (positive_probability - matched_negative_mean_probability)
        - worst 10 / best 10 positives by that difference

DIAGNOSTIC ONLY. Never trains anything, never modifies predictions/manifest/split/checkpoints of
the run(s) it evaluates. Read-only with respect to test_predictions_resumable.csv, manifest.csv,
split_manifest.csv for every run-dir it's pointed at.

REUSED, UNMODIFIED, from diagnose_cross_sample_shift.py (this session's already-validated code):
    load_split_manifest, load_manifest_by_filename, load_predictions,
    cross_validate_predictions_against_manifest, load_raw_patch_stats (the resumable,
    checkpointed raw-mean/raw-std loader -- reused here for ANY run's own directory, giving each
    run its own independent checkpoint at <run-dir>/diagnostics/raw_patch_stats_checkpoint.npz),
    TEST_SAMPLES

Usage (single run):
    python scripts/benchmark_v2/diagnose_matched_contrast.py \\
        --run-dir results/exp05_training_dataset/run02_20samples/run08_patch_instance_norm \\
        --stage1-root results/exp05_training_dataset/run02_20samples

Usage (Run08 vs Run09 side-by-side comparison, once both have test_predictions_resumable.csv):
    python scripts/benchmark_v2/diagnose_matched_contrast.py \\
        --run-dir results/exp05_training_dataset/run02_20samples/run08_patch_instance_norm \\
        --compare-run-dir results/exp05_training_dataset/run02_20samples/run09_contrast_aug_focal_robust_eps \\
        --stage1-root results/exp05_training_dataset/run02_20samples
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
from scipy.stats import pearsonr, spearmanr

from diagnose_cross_sample_shift import (
    load_split_manifest,
    load_manifest_by_filename,
    load_predictions,
    cross_validate_predictions_against_manifest,
    load_raw_patch_stats,
    TEST_SAMPLES,
)

K_NEAREST_DEFAULT = 20
WORST_BEST_N_DEFAULT = 10
MATCHED_CONTRAST_CHECKPOINT_SUBDIR = "diagnostics"


# --------------------------------------------------------------------------
# Nearest-neighbor-by-std matching
# --------------------------------------------------------------------------

def find_k_nearest_by_value(sorted_values, query_value, k):
    """Returns the indices (into sorted_values) of the k entries whose value is closest to
    query_value, via a two-pointer expansion around the insertion point (O(k) per query after an
    O(log n) search -- fine for n~16,476 negatives x 111 positives = ~1.8M comparisons total)."""
    n = len(sorted_values)
    k = min(k, n)
    pos = int(np.searchsorted(sorted_values, query_value))
    lo, hi = pos - 1, pos
    chosen = []
    while len(chosen) < k:
        left_ok = lo >= 0
        right_ok = hi < n
        if not left_ok and not right_ok:
            break
        if left_ok and (not right_ok or (query_value - sorted_values[lo]) <= (sorted_values[hi] - query_value)):
            chosen.append(lo)
            lo -= 1
        else:
            chosen.append(hi)
            hi += 1
    return chosen


def compute_matched_contrast(predictions, raw_means, raw_stds, k):
    """predictions/raw_means/raw_stds are order-aligned arrays/lists over the FULL test set for
    one run (as produced by load_raw_patch_stats). Matching is pooled across the whole test set
    (not restricted per biological sample), matching the Run08 numbers you reported (which were
    global test-set statistics)."""
    labels = np.array([p["label"] for p in predictions], dtype=np.float32)
    probs = np.array([p["probability"] for p in predictions], dtype=np.float64)

    neg_idx = np.where(labels == 0.0)[0]
    pos_idx = np.where(labels == 1.0)[0]
    if len(neg_idx) == 0 or len(pos_idx) == 0:
        raise ValueError("BLOCKED: need at least one positive and one negative to compute matched contrast.")

    neg_stds = raw_stds[neg_idx]
    order = np.argsort(neg_stds)
    neg_idx_sorted = neg_idx[order]
    neg_stds_sorted = neg_stds[order]

    per_positive = []
    for i in pos_idx:
        query_std = raw_stds[i]
        nearest_local = find_k_nearest_by_value(neg_stds_sorted, query_std, k)
        matched_global_idx = neg_idx_sorted[nearest_local]
        matched_probs = probs[matched_global_idx]
        matched_stds = raw_stds[matched_global_idx]
        pos_prob = probs[i]
        entry = {
            "filename": predictions[i]["filename"],
            "sample_id": predictions[i]["sample_id"],
            "positive_probability": float(pos_prob),
            "positive_raw_mean": float(raw_means[i]),
            "positive_raw_std": float(raw_stds[i]),
            "k_matched": len(matched_global_idx),
            "matched_negative_mean_probability": float(matched_probs.mean()),
            "matched_negative_median_probability": float(np.median(matched_probs)),
            "matched_negative_std_range": [float(matched_stds.min()), float(matched_stds.max())],
            "diff_vs_matched_mean": float(pos_prob - matched_probs.mean()),
            "diff_vs_matched_median": float(pos_prob - float(np.median(matched_probs))),
            "beats_matched_mean": bool(pos_prob > matched_probs.mean()),
            "beats_matched_median": bool(pos_prob > np.median(matched_probs)),
        }
        per_positive.append(entry)

    diffs_mean = np.array([e["diff_vs_matched_mean"] for e in per_positive])
    diffs_median = np.array([e["diff_vs_matched_median"] for e in per_positive])
    n_pos = len(per_positive)

    summary = {
        "k_nearest": k,
        "n_positives": n_pos,
        "n_negatives_pool": int(len(neg_idx)),
        "mean_positive_probability": float(np.mean([e["positive_probability"] for e in per_positive])),
        "mean_matched_negative_probability": float(np.mean([e["matched_negative_mean_probability"] for e in per_positive])),
        "median_positive_probability": float(np.median([e["positive_probability"] for e in per_positive])),
        "median_matched_negative_probability": float(np.median([e["matched_negative_median_probability"] for e in per_positive])),
        "pct_positive_beats_matched_mean": 100.0 * sum(e["beats_matched_mean"] for e in per_positive) / n_pos,
        "pct_positive_beats_matched_median": 100.0 * sum(e["beats_matched_median"] for e in per_positive) / n_pos,
        "diff_vs_matched_mean_distribution": {
            "mean": float(diffs_mean.mean()), "median": float(np.median(diffs_mean)),
            "std": float(diffs_mean.std()), "min": float(diffs_mean.min()), "max": float(diffs_mean.max()),
            "p25": float(np.percentile(diffs_mean, 25)), "p75": float(np.percentile(diffs_mean, 75)),
        },
        "diff_vs_matched_median_distribution": {
            "mean": float(diffs_median.mean()), "median": float(np.median(diffs_median)),
            "std": float(diffs_median.std()), "min": float(diffs_median.min()), "max": float(diffs_median.max()),
        },
    }

    ranked = sorted(per_positive, key=lambda e: e["diff_vs_matched_mean"])
    worst = ranked[:WORST_BEST_N_DEFAULT]
    best = list(reversed(ranked[-WORST_BEST_N_DEFAULT:]))

    return summary, per_positive, worst, best


def compute_probability_std_correlations(predictions, raw_stds):
    probs = np.array([p["probability"] for p in predictions], dtype=np.float64)
    pearson_r, _ = pearsonr(raw_stds, probs)
    spearman_r, _ = spearmanr(raw_stds, probs)
    return float(pearson_r), float(spearman_r)


# --------------------------------------------------------------------------
# Per-run pipeline
# --------------------------------------------------------------------------

def run_matched_contrast_for_run_dir(run_dir, stage1_root, split_manifest_path, manifest_csv_path,
                                      predictions_csv_path, patch_size, k):
    """Loads + validates predictions for ONE run, reuses load_raw_patch_stats (resumable,
    checkpointed, UNMODIFIED) to get raw_means/raw_stds for that run's own test predictions, then
    computes the matched-contrast diagnostic. Returns everything needed for printing/saving."""
    for required in [split_manifest_path, manifest_csv_path, predictions_csv_path]:
        if not required.exists():
            raise SystemExit(f"BLOCKED: required file does not exist: {required}")

    split_assignment = load_split_manifest(split_manifest_path)
    manifest_by_filename = load_manifest_by_filename(manifest_csv_path)
    predictions = load_predictions(predictions_csv_path)
    cross_validate_predictions_against_manifest(predictions, manifest_by_filename, split_assignment)
    print(f"[{run_dir.name}] Loaded and cross-validated {len(predictions)} prediction rows.")

    diagnostics_dir = run_dir / MATCHED_CONTRAST_CHECKPOINT_SUBDIR
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    # load_raw_patch_stats is fully generic over (predictions, out_dir) -- reusing it here gives
    # this run its own independent, resumable, fingerprinted checkpoint, same formula/discipline
    # as everywhere else in this pipeline, with ZERO new caching logic written for this script.
    raw_means, raw_stds = load_raw_patch_stats(predictions, patch_size, diagnostics_dir)
    print(f"[{run_dir.name}] Raw patch stats ready ({len(raw_means)} patches).")

    pearson_r, spearman_r = compute_probability_std_correlations(predictions, raw_stds)
    summary, per_positive, worst, best = compute_matched_contrast(predictions, raw_means, raw_stds, k)
    summary["pearson_r_probability_vs_raw_std"] = pearson_r
    summary["spearman_r_probability_vs_raw_std"] = spearman_r

    return summary, per_positive, worst, best, predictions, raw_means, raw_stds


def print_summary(label, summary):
    print(f"\n--- Matched-contrast summary: {label} ---")
    print(f"  n_positives={summary['n_positives']}  n_negatives_pool={summary['n_negatives_pool']}  k={summary['k_nearest']}")
    print(f"  mean positive prob      = {summary['mean_positive_probability']:.4f}")
    print(f"  mean matched-neg prob   = {summary['mean_matched_negative_probability']:.4f}")
    print(f"  median positive prob     = {summary['median_positive_probability']:.4f}")
    print(f"  median matched-neg prob = {summary['median_matched_negative_probability']:.4f}")
    print(f"  positive beats matched-neg MEAN:   {summary['pct_positive_beats_matched_mean']:.2f}%")
    print(f"  positive beats matched-neg MEDIAN: {summary['pct_positive_beats_matched_median']:.2f}%")
    print(f"  Pearson r(probability, raw_std)  = {summary['pearson_r_probability_vs_raw_std']:.4f}")
    print(f"  Spearman r(probability, raw_std) = {summary['spearman_r_probability_vs_raw_std']:.4f}")


def save_outputs(run_dir, summary, per_positive, worst, best):
    diagnostics_dir = run_dir / MATCHED_CONTRAST_CHECKPOINT_SUBDIR
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    json_path = diagnostics_dir / "matched_contrast_diagnostic.json"
    csv_path = diagnostics_dir / "matched_contrast_per_positive.csv"
    result = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "summary": summary,
        "worst_10_by_diff_vs_matched_mean": worst,
        "best_10_by_diff_vs_matched_mean": best,
    }
    json_path.write_text(json.dumps(result, indent=2))
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(per_positive[0].keys()) if per_positive else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_positive)
    print(f"Saved: {json_path}\nSaved: {csv_path}")
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--compare-run-dir", default=None,
                         help="Optional second run directory (e.g. Run09) to compute the same "
                              "diagnostic for and print a direct side-by-side comparison table.")
    parser.add_argument("--stage1-root", required=True)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--manifest-csv", default=None)
    parser.add_argument("--k-nearest", type=int, default=K_NEAREST_DEFAULT)
    parser.add_argument("--patch-size", type=int, default=32)
    args = parser.parse_args()

    stage1_root = Path(args.stage1_root)
    split_manifest_path = Path(args.split_manifest) if args.split_manifest else stage1_root / "split_manifest.csv"
    manifest_csv_path = Path(args.manifest_csv) if args.manifest_csv else stage1_root / "manifest.csv"

    run_dir = Path(args.run_dir)
    predictions_csv_path = run_dir / "test_predictions_resumable.csv"
    summary1, per_pos1, worst1, best1, preds1, means1, stds1 = run_matched_contrast_for_run_dir(
        run_dir, stage1_root, split_manifest_path, manifest_csv_path, predictions_csv_path,
        args.patch_size, args.k_nearest,
    )
    print_summary(run_dir.name, summary1)
    save_outputs(run_dir, summary1, per_pos1, worst1, best1)

    if args.compare_run_dir:
        compare_run_dir = Path(args.compare_run_dir)
        compare_predictions_csv_path = compare_run_dir / "test_predictions_resumable.csv"
        summary2, per_pos2, worst2, best2, preds2, means2, stds2 = run_matched_contrast_for_run_dir(
            compare_run_dir, stage1_root, split_manifest_path, manifest_csv_path,
            compare_predictions_csv_path, args.patch_size, args.k_nearest,
        )
        print_summary(compare_run_dir.name, summary2)
        save_outputs(compare_run_dir, summary2, per_pos2, worst2, best2)

        print(f"\n{'='*70}\n{run_dir.name} vs {compare_run_dir.name}\n{'='*70}")

        def fmt_delta(a, b, higher_is_better=True):
            delta = b - a
            arrow = "better" if (delta > 0) == higher_is_better else ("worse" if delta != 0 else "unchanged")
            return f"{a:.4f} -> {b:.4f}  ({delta:+.4f}, {arrow})"

        print(f"Pearson |r(prob, raw_std)|  : {fmt_delta(abs(summary1['pearson_r_probability_vs_raw_std']), abs(summary2['pearson_r_probability_vs_raw_std']), higher_is_better=False)}")
        print(f"Spearman |r(prob, raw_std)| : {fmt_delta(abs(summary1['spearman_r_probability_vs_raw_std']), abs(summary2['spearman_r_probability_vs_raw_std']), higher_is_better=False)}")
        print(f"Matched positive > neg mean  : {fmt_delta(summary1['pct_positive_beats_matched_mean'], summary2['pct_positive_beats_matched_mean'])}")
        print(f"Matched positive > neg median: {fmt_delta(summary1['pct_positive_beats_matched_median'], summary2['pct_positive_beats_matched_median'])}")

        print("\nNOTE: this table covers ONLY the matched-contrast/correlation numbers this "
              "script computes. ROC-AUC/PR-AUC/balanced-accuracy/F1/precision/recall come from "
              "diagnose_run08_test.py's and diagnose_run09_test.py's own diagnostic JSON files "
              "-- combine both before writing a verdict. This script deliberately does NOT "
              "compute or print a SUCCESS/PARTIAL SUCCESS/NO IMPROVEMENT/REGRESSION verdict "
              "itself, since that requires the full metric set, not just this one (critical) "
              "diagnostic.")


if __name__ == "__main__":
    main()
