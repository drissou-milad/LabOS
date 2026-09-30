# Benchmark v2 — distance-gated evaluation (TP / FP / FN / precision / recall / F1)

Entirely separate scoring methodology from `scripts/labos_diagnostic_benchmark.py`
(Benchmark v1). Does not modify, import from, or write to anything belonging to v1.
`results/benchmark_baseline_v1.csv` is untouched — `run_benchmark_v2.py` refuses to write to
any file named `benchmark_baseline_v1.csv`, in addition to refusing to overwrite any existing
output file.

## Why this exists

Benchmark v1's matching (`linear_sum_assignment` with no distance cutoff) always pairs every
GT point with *some* candidate whenever candidates exist, so a true miss never shows up as a
drop in "coverage" — it only inflates the mean/tail distance. Benchmark v2 adds a hard distance
gate (`d_max=15px` by default) so misses (FN), hallucinated candidates (FP), and genuine
matches (TP) are counted as distinct, honest outcomes.

## Method (see `matching_v2.py` docstring for full detail)

1. Build the true distance matrix (GT × candidates).
2. Cap distances above `d_max` at a large penalty, then run Hungarian assignment on the
   capped matrix — this keeps the solver from being incentivized to trade a valid nearby match
   for an invalid distant one, without ever allowing an invalid (`>d_max`) pair to count.
3. A solved pair counts as **TP** only if its *true* distance is `<= d_max`.
4. `FN = n_gt - TP`, `FP = n_pred - TP` (correctly counts both "assigned but too far" and
   "never assigned because the other set was larger" — no double counting).

Verified against a hand-computed synthetic case (1 TP, 1 FN, 2 FP, precision=1/3, recall=0.5)
and edge cases (empty GT, empty predictions, both empty) — see verification notes at the
bottom of this file.

## Metrics reported (per sample, micro-averaged across all frames)

`tp`, `fp`, `fn`, `precision`, `recall`, `f1`, `mean_error_px_tp_only`,
`median_error_px_tp_only`, `frac_le_5px_tp_only`, `frac_le_10px_tp_only`,
`frac_le_15px_tp_only` — the last is 1.0 by construction when `d_max=15` (included for shape
consistency if you rerun with a different `--d-max`). A `_frames.csv` sibling file gives the
same breakdown per frame.

Micro-averaging (summing TP/FP/FN across all of a sample's frames before computing
precision/recall/F1 once) is used rather than averaging per-frame rates, because per-frame
precision/recall is undefined on frames with zero GT or zero predictions.

## Commands

```bash
export BIOHUB_DATASET_PATH=/path/to/biohub-cell-tracking-during-development

# Smoke test
python scripts/benchmark_v2/run_benchmark_v2.py \
    --out results/benchmark_v2/benchmark_v2_smoke.csv \
    --limit-samples 3

# Full run (baseline detector, unmodified, d_max=15px)
python scripts/benchmark_v2/run_benchmark_v2.py \
    --out results/benchmark_v2/benchmark_v2_baseline_detector.csv
```

`--d-max` and `--detector` are available if you want to sweep the gate distance or (once
extended) score a different detector under this same stricter methodology — see
`build_detector()` in `run_benchmark_v2.py` for how to wire in
`AdaptiveCellDetector` from Experiment #1 later.

## Interpretation notes

- Report v1 and v2 **side by side**, not as a replacement for each other — they answer
  different questions. v1: "how far off are matched points, on average, given at least one
  candidate exists nearby." v2: "how many real cells did the detector actually find, miss, or
  hallucinate."
- v2's `recall` is the metric that actually behaves like recall (unlike v1's uncapped
  "coverage", which is structurally close to 100% almost regardless of true detector
  performance, per the analysis in `LabOS_Detection_Failure_Analysis_v1.md`, §3).
- `mean/median_error_px_tp_only` is not directly comparable to v1's `sample_mean_error_px` —
  v2's error stats exclude every FN/FP by construction, so v2's error numbers will typically
  look *better* than v1's on the same detector, for a reason that has nothing to do with the
  detector improving: v1 is including some genuinely-bad forced matches in its average that v2
  correctly excludes as non-matches. This is expected and should be stated explicitly if both
  numbers appear in the same report, to avoid it reading like an unexplained improvement.

## Verification performed before handoff (no real BioHub data available in this environment)

- `matching_v2.gated_match` hand-verified against a manually constructed case with a known
  correct answer (1 TP at 3px, 1 FN, 2 FP, precision=1/3, recall=0.5), plus empty-GT,
  empty-prediction, and both-empty edge cases.
- `run_benchmark_v2.py`'s full pipeline (dataset → detector → gated matching → CSV) run
  end-to-end against a synthetic fake dataset with GT placed exactly at 3 known blob centers
  across 2 frames — produced the expected 6/6 TP, 0 FP, 0 FN, precision=recall=F1=1.0,
  ~0px error.
- Both guardrails (refuse to overwrite an existing `--out`, refuse to ever write to a file
  named `benchmark_baseline_v1.csv`) triggered correctly.
- **Not yet run against the real BioHub dataset** — not available in this environment. Run the
  commands above in your environment with `BIOHUB_DATASET_PATH` set.
