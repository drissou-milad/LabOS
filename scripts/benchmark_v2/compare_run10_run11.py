"""
Experiment #5: Run10-vs-Run11 comparison table.

Reads ONLY files already produced by train_exp05_run10.py / train_exp05_run11.py,
diagnose_run10_test.py / diagnose_run11_test.py, and diagnose_matched_contrast.py /
diagnose_cross_sample_shift.py (run separately against each run-dir, unmodified, per their own
"works against any run-dir" design). Computes and fabricates NOTHING -- if a required input file
is missing, this script stops and tells you which diagnostic to run first, rather than silently
omitting a row or guessing a number.

DIAGNOSTIC ONLY. Never trains anything, never re-runs inference, never touches a checkpoint.

Usage:
    python scripts/benchmark_v2/compare_run10_run11.py \\
        --run10-dir results/exp05_training_dataset/run02_20samples/run10_no_classifier_relu \\
        --run11-dir results/exp05_training_dataset/run02_20samples/run11_sample_robust_norm
"""

import argparse
import csv
import json
from pathlib import Path


TEST_SAMPLE_ORDER = ["44b6_0b24845f", "44b6_1d530831", "44b6_33b596bf"]


def require(path, produced_by):
    if not path.exists():
        raise SystemExit(f"BLOCKED: required file not found: {path}\n"
                          f"  -> produce it first with: {produced_by}")
    return path


def load_json(path):
    return json.loads(path.read_text())


def best_val_row(training_metrics_csv):
    """Best epoch BY val_pr_auc -- same checkpoint-selection metric used to pick model.pth, so
    this is the val performance of the ACTUAL checkpoint being evaluated on test, not just the
    last epoch trained."""
    with open(training_metrics_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"BLOCKED: {training_metrics_csv} is empty.")
    best = max(rows, key=lambda r: float(r["val_pr_auc"]))
    return {"epoch": int(best["epoch"]),
            "val_roc_auc": float(best["val_roc_auc"]),
            "val_pr_auc": float(best["val_pr_auc"])}


def per_sample_roc_auc(cross_sample_shift_json):
    """Pulls per-sample ROC-AUC out of diagnose_cross_sample_shift.py's own output -- the single
    canonical source for this number for any run, generic and unmodified across Run08-11."""
    out = {}
    per_sample = cross_sample_shift_json.get("per_sample") or cross_sample_shift_json.get("PER_SAMPLE")
    if per_sample is None:
        # fall back to the same nested shape used in the audited example package
        # (per-sample entries keyed directly at top level under known sample ids)
        for sid in TEST_SAMPLE_ORDER:
            if sid in cross_sample_shift_json:
                out[sid] = cross_sample_shift_json[sid].get("roc_auc")
        return out
    for sid, entry in per_sample.items():
        out[sid] = entry.get("roc_auc")
    return out


def fmt(x, nd=4):
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def pct_change(run10_val, run11_val):
    if run10_val is None or run11_val is None:
        return "n/a"
    try:
        r10, r11 = float(run10_val), float(run11_val)
    except (TypeError, ValueError):
        return "n/a"
    delta = r11 - r10
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run10-dir", required=True)
    parser.add_argument("--run11-dir", required=True)
    parser.add_argument("--out-path", default=None,
                         help="Default: <run11-dir>/run10_vs_run11_comparison.md")
    args = parser.parse_args()

    run10_dir = Path(args.run10_dir)
    run11_dir = Path(args.run11_dir)

    # ---- required inputs (each with the exact command that produces it) ----
    run10_test_json = require(run10_dir / "run10_test_diagnostic.json",
        f"python scripts/benchmark_v2/diagnose_run10_test.py --run-dir {run10_dir} --stage1-root <stage1-root> --overwrite")
    run11_test_json = require(run11_dir / "run11_test_diagnostic.json",
        f"python scripts/benchmark_v2/diagnose_run11_test.py --run-dir {run11_dir} --stage1-root <stage1-root> --overwrite")
    run10_train_metrics = require(run10_dir / "training_metrics.csv", "train_exp05_run10.py (already run)")
    run11_train_metrics = require(run11_dir / "training_metrics.csv", "train_exp05_run11.py (already run)")
    run10_matched = require(run10_dir / "diagnostics" / "matched_contrast_diagnostic.json",
        f"python scripts/benchmark_v2/diagnose_matched_contrast.py --run-dir {run10_dir} --stage1-root <stage1-root>")
    run11_matched = require(run11_dir / "diagnostics" / "matched_contrast_diagnostic.json",
        f"python scripts/benchmark_v2/diagnose_matched_contrast.py --run-dir {run11_dir} --stage1-root <stage1-root>")
    run10_shift = require(run10_dir / "diagnostics" / "cross_sample_shift_diagnostic.json",
        f"python scripts/benchmark_v2/diagnose_cross_sample_shift.py --run-dir {run10_dir} --stage1-root <stage1-root> --overwrite")
    run11_shift = require(run11_dir / "diagnostics" / "cross_sample_shift_diagnostic.json",
        f"python scripts/benchmark_v2/diagnose_cross_sample_shift.py --run-dir {run11_dir} --stage1-root <stage1-root> --overwrite")

    r10_test = load_json(run10_test_json)
    r11_test = load_json(run11_test_json)
    r10_val = best_val_row(run10_train_metrics)
    r11_val = best_val_row(run11_train_metrics)
    r10_matched = load_json(run10_matched)["summary"]
    r11_matched = load_json(run11_matched)["summary"]
    r10_shift = load_json(run10_shift)
    r11_shift = load_json(run11_shift)

    r10_per_sample = per_sample_roc_auc(r10_shift)
    r11_per_sample = per_sample_roc_auc(r11_shift)

    def spread(per_sample_dict):
        vals = [v for v in per_sample_dict.values() if v is not None]
        return (max(vals) - min(vals)) if len(vals) >= 2 else None

    rows = []
    rows.append(("val_PR_AUC (best checkpoint, val_pr_auc-selected)", r10_val["val_pr_auc"], r11_val["val_pr_auc"]))
    rows.append(("val_ROC_AUC (best checkpoint)", r10_val["val_roc_auc"], r11_val["val_roc_auc"]))
    rows.append(("test_PR_AUC", r10_test.get("pr_auc"), r11_test.get("pr_auc")))
    rows.append(("test_ROC_AUC", r10_test.get("roc_auc"), r11_test.get("roc_auc")))
    rows.append(("matched_contrast_win_rate_%% (vs matched-negative MEAN)",
                 r10_matched.get("pct_positive_beats_matched_mean"),
                 r11_matched.get("pct_positive_beats_matched_mean")))
    rows.append(("|Pearson r(probability, raw_std)|",
                 abs(r10_matched["pearson_r_probability_vs_raw_std"]) if r10_matched.get("pearson_r_probability_vs_raw_std") is not None else None,
                 abs(r11_matched["pearson_r_probability_vs_raw_std"]) if r11_matched.get("pearson_r_probability_vs_raw_std") is not None else None))
    rows.append(("|Spearman r(probability, raw_std)|",
                 abs(r10_matched["spearman_r_probability_vs_raw_std"]) if r10_matched.get("spearman_r_probability_vs_raw_std") is not None else None,
                 abs(r11_matched["spearman_r_probability_vs_raw_std"]) if r11_matched.get("spearman_r_probability_vs_raw_std") is not None else None))
    for sid in TEST_SAMPLE_ORDER:
        rows.append((f"{sid}_ROC_AUC", r10_per_sample.get(sid), r11_per_sample.get(sid)))
    rows.append(("per_sample_ROC_AUC_spread (max-min across 3 test samples)",
                 spread(r10_per_sample), spread(r11_per_sample)))

    lines = []
    lines.append("# Run10 vs Run11 comparison\n")
    lines.append("| Metric | Run10 | Run11 | Change |")
    lines.append("|---|---|---|---|")
    for name, r10_v, r11_v in rows:
        lines.append(f"| {name} | {fmt(r10_v)} | {fmt(r11_v)} | {pct_change(r10_v, r11_v)} |")

    # ---- success-criteria checklist, evaluated mechanically against the 8 criteria specified
    # for Run11, WITHOUT editorializing about whether the run "succeeded" -- that judgment is
    # left to the person reading the table, per "Do NOT claim Run11 is better before actually
    # training and evaluating it."
    lines.append("\n## Success-criteria checklist (mechanical, not a verdict)\n")

    def check(label, r10_v, r11_v, better):
        if r10_v is None or r11_v is None:
            lines.append(f"- [ ] {label}: n/a (missing value)")
            return
        ok = better(float(r10_v), float(r11_v))
        lines.append(f"- [{'x' if ok else ' '}] {label}: Run10={fmt(r10_v)} -> Run11={fmt(r11_v)}")

    check("1. test PR-AUC improves over Run10", r10_test.get("pr_auc"), r11_test.get("pr_auc"), lambda a, b: b > a)
    check("2. test ROC-AUC improves over Run10", r10_test.get("roc_auc"), r11_test.get("roc_auc"), lambda a, b: b > a)
    check("3. matched-contrast win rate increases substantially above 54.05%",
          r10_matched.get("pct_positive_beats_matched_mean"), r11_matched.get("pct_positive_beats_matched_mean"),
          lambda a, b: b > a + 5.0)
    check("4. |Pearson r| decreases from ~0.367",
          abs(r10_matched["pearson_r_probability_vs_raw_std"]) if r10_matched.get("pearson_r_probability_vs_raw_std") is not None else None,
          abs(r11_matched["pearson_r_probability_vs_raw_std"]) if r11_matched.get("pearson_r_probability_vs_raw_std") is not None else None,
          lambda a, b: b < a)
    check("5. 44b6_0b24845f ROC-AUC improves substantially from ~0.142",
          r10_per_sample.get("44b6_0b24845f"), r11_per_sample.get("44b6_0b24845f"), lambda a, b: b > a + 0.1)
    check("7. per-sample ROC-AUC spread decreases",
          spread(r10_per_sample), spread(r11_per_sample), lambda a, b: b < a)
    check("8. improvement also visible on validation PR-AUC (not test-only)",
          r10_val["val_pr_auc"], r11_val["val_pr_auc"], lambda a, b: b > a)
    lines.append("\n(Criteria 6 -- \"more consistent across samples\" -- is qualitative; read "
                 "the three per-sample ROC-AUC rows above directly rather than relying on a "
                 "single pass/fail check.)")
    lines.append("\nRemember the stated bar: do NOT call Run11 successful merely because pooled "
                 "ROC-AUC increased, and do NOT call it successful if it improves one metric "
                 "while making the others in this table worse.")

    out_path = Path(args.out_path) if args.out_path else run11_dir / "run10_vs_run11_comparison.md"
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
