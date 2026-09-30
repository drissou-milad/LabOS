"""
Compare two Benchmark v2 (distance-gated) per-sample CSVs: dataset-wide aggregate deltas
(TP/FP/FN/precision/recall/F1/mean+median TP error) plus per-sample improved/worsened/unchanged
counts for precision, recall, and F1 independently.

New script. Read-only: does not modify either input CSV, run_benchmark_v2.py, matching_v2.py,
or scripts/compare_benchmarks.py (which is v1's mean/median-error schema and cannot score TP/
FP/FN -- this is a deliberately separate tool, not a modification of that one).

Usage:
    python scripts/benchmark_v2/compare_v2_aggregate.py \
        --baseline-csv results/benchmark_v2/benchmark_v2_baseline_detector.csv \
        --candidate-csv results/exp01_adaptive_threshold/benchmark_v2_adaptive.csv \
        --out-dir results/exp01_adaptive_threshold/comparison_v2

Produces:
    comparison_v2_summary.txt   dataset-wide aggregate before/after + deltas
    per_sample_v2_delta.csv     per-sample precision/recall/F1 before/after + delta + direction
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 and not (np.isnan(precision) or np.isnan(recall))
          else float("nan"))
    return precision, recall, f1


def weighted_mean_tp_error(df):
    """Exact dataset-wide mean, reconstructed from per-sample means weighted by each sample's
    TP count. This IS exact (mean-of-means weighted by count = true pooled mean). Dataset-wide
    MEDIAN cannot be exactly reconstructed this way -- only from the raw pooled distances, which
    run_benchmark_v2_adaptive.py's own *_summary.txt writes directly. Prefer that summary's
    median over anything derived here if it's available."""
    valid = df.dropna(subset=["mean_error_px_tp_only"])
    if valid["tp"].sum() == 0:
        return float("nan")
    return float((valid["mean_error_px_tp_only"] * valid["tp"]).sum() / valid["tp"].sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline-summary-txt", default=None,
                         help="Optional: the *_summary.txt from the baseline run, for an exact "
                              "(not weighted-mean-reconstructed) mean/median TP error comparison")
    parser.add_argument("--candidate-summary-txt", default=None,
                         help="Optional: the *_summary.txt from the candidate run (written "
                              "automatically by run_benchmark_v2_adaptive.py)")
    args = parser.parse_args()

    base = pd.read_csv(args.baseline_csv)
    cand = pd.read_csv(args.candidate_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    common = sorted(set(base["sample"]) & set(cand["sample"]))
    if len(common) < len(base) or len(common) < len(cand):
        print(f"NOTE: comparing on {len(common)} samples common to both files "
              f"(baseline has {len(base)}, candidate has {len(cand)}).")
    base_c = base[base["sample"].isin(common)].set_index("sample")
    cand_c = cand[cand["sample"].isin(common)].set_index("sample")

    # ---- dataset-wide aggregate (sum counts, recompute rates) ----
    def agg(df):
        tp, fp, fn = int(df["tp"].sum()), int(df["fp"].sum()), int(df["fn"].sum())
        p, r, f1 = precision_recall_f1(tp, fp, fn)
        return {
            "n_gt": int(df["n_gt_total"].sum()), "n_pred": int(df["n_pred_total"].sum()),
            "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1,
            "mean_tp_error_weighted": weighted_mean_tp_error(df),
        }

    base_agg = agg(base_c)
    cand_agg = agg(cand_c)

    lines = ["=" * 70, "Benchmark v2 — Baseline vs. Experiment #1 (adaptive threshold)", "=" * 70, ""]
    for label, a in [("Baseline (frozen)", base_agg), ("Candidate (adaptive)", cand_agg)]:
        lines.append(f"-- {label} --")
        lines.append(f"  GT = {a['n_gt']}")
        lines.append(f"  predictions = {a['n_pred']}")
        lines.append(f"  TP = {a['tp']}")
        lines.append(f"  FP = {a['fp']}")
        lines.append(f"  FN = {a['fn']}")
        lines.append(f"  precision = {a['precision']:.7g}")
        lines.append(f"  recall = {a['recall']:.7g}")
        lines.append(f"  F1 = {a['f1']:.7g}")
        lines.append(f"  mean TP error (weighted from per-sample means) = {a['mean_tp_error_weighted']:.3f} px")
        lines.append("")

    lines.append("-- Deltas (candidate - baseline) --")
    lines.append(f"  delta TP = {cand_agg['tp'] - base_agg['tp']:+d}")
    lines.append(f"  delta FP = {cand_agg['fp'] - base_agg['fp']:+d}")
    lines.append(f"  delta FN = {cand_agg['fn'] - base_agg['fn']:+d}")
    lines.append(f"  delta precision = {cand_agg['precision'] - base_agg['precision']:+.5f}")
    lines.append(f"  delta recall = {cand_agg['recall'] - base_agg['recall']:+.5f}")
    lines.append(f"  delta F1 = {cand_agg['f1'] - base_agg['f1']:+.5f}")
    lines.append("")
    lines.append(
        "NOTE: these dataset-wide mean-TP-error and precision/recall/F1 numbers are recomputed "
        "here from the two per-sample CSVs (summed TP/FP/FN, and TP-count-weighted mean error, "
        "both of which are exact). If a baseline *_summary.txt or an externally reported "
        "baseline aggregate exists, cross-check against it -- if the baseline number wasn't "
        "computed the same way (e.g. an unweighted average of per-sample means instead of a "
        "true pooled mean), the two numbers can differ slightly for reasons that have nothing "
        "to do with the candidate detector."
    )

    # ---- per-sample precision/recall/F1 deltas + improved/worsened/unchanged counts ----
    per_sample = pd.DataFrame({
        "sample": common,
        "baseline_precision": base_c["precision"].values,
        "candidate_precision": cand_c["precision"].values,
        "baseline_recall": base_c["recall"].values,
        "candidate_recall": cand_c["recall"].values,
        "baseline_f1": base_c["f1"].values,
        "candidate_f1": cand_c["f1"].values,
    })
    for metric in ["precision", "recall", "f1"]:
        per_sample[f"delta_{metric}"] = per_sample[f"candidate_{metric}"] - per_sample[f"baseline_{metric}"]

    per_sample.to_csv(out_dir / "per_sample_v2_delta.csv", index=False)

    lines.append("")
    lines.append("-- Per-sample improved / worsened / unchanged counts --")
    for metric in ["precision", "recall", "f1"]:
        d = per_sample[f"delta_{metric}"].dropna()
        improved = int((d > 0).sum())
        worsened = int((d < 0).sum())
        unchanged = int((d == 0).sum())
        lines.append(f"  {metric:10s}: improved={improved:3d}  worsened={worsened:3d}  "
                     f"unchanged={unchanged:3d}  (n={len(d)}, mean delta={d.mean():+.5f})")

    summary_text = "\n".join(lines)
    (out_dir / "comparison_v2_summary.txt").write_text(summary_text)
    print(summary_text)
    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()
