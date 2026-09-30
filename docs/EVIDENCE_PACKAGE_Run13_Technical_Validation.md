# Run13 Technical Validation Evidence Package

**Project:** LabOS
**Checkpoint:** v0.7
**Evaluation:** Run13 held-out test evaluation
**Status:** Technical research validation package

## 1. Purpose

This document records the technical evidence for the Run13 held-out evaluation of the LabOS cell detector. It preserves the evaluation methodology, dataset composition, metrics, normalization audit, spatial/temporal audit, and limitations.

## 2. Evaluation Configuration

- Evaluation directory: `results/exp05_training_dataset/run02_20samples/run13_test_eval`
- Prediction file: `test_predictions_run13.csv`
- Checkpoint SHA-256 prefix: `55e1ee93`
- Test samples:
  - `44b6_0b24845f`
  - `44b6_1d530831`
  - `44b6_33b596bf`
- Total patches: **16,587**
- Positive patches: **111**
- Negative patches: **16,476**
- Raw patch size: **32x32**
- Evaluation crop: `[8:24,8:24]`
- Resize: **2.0x**, interpolation order 1
- Model: `CellCNNNoClassifierReLU`
- Centering tolerance: **1.5 px**

The detector uses maximum-intensity projection (MIP). The source volume shape is `(100,64,256,256)`, producing a 256x256 frame from the 64-slice Z-stack.

## 3. Dataset Composition

| Sample | Total | Positive | Negative |
|---|---:|---:|---:|
| `44b6_0b24845f` | 2,068 | 4 | 2,064 |
| `44b6_1d530831` | 9,763 | 95 | 9,668 |
| `44b6_33b596bf` | 4,756 | 12 | 4,744 |
| **Total** | **16,587** | **111** | **16,476** |

Positive prevalence is approximately **0.669%**, making precision particularly sensitive to false positives.

## 4. Overall Run13 Metrics

At the predefined threshold **0.50**:

| Metric | Value |
|---|---:|
| ROC-AUC | **0.656016** |
| PR-AUC | **0.014390** |
| Accuracy | **0.543076** |
| Balanced Accuracy | **0.595507** |
| Precision | **0.009459** |
| Recall | **0.648649** |
| F1 | **0.018646** |
| TP | 72 |
| FP | 7,540 |
| FN | 39 |
| TN | 8,936 |

### Confusion Matrix

```text
[[8936, 7540],
 [  39,   72]]
```

The results demonstrate measurable ranking/discriminative signal, but the threshold-0.50 operating point produces a very high false-positive count and should not be interpreted as a reliable production detector.
## 5. Per-Sample Discrimination

| Sample | ROC-AUC | PR-AUC | Positive mean | Negative mean |
|---|---:|---:|---:|---:|
| `44b6_0b24845f` | 0.335514 | 0.001806 | 0.4479 | 0.4706 |
| `44b6_1d530831` | 0.651407 | 0.022004 | 0.5289 | 0.4965 |
| `44b6_33b596bf` | 0.721684 | 0.007243 | 0.5434 | 0.4983 |

The sample-level results are heterogeneous. In particular, `44b6_0b24845f` does not show positive ranking separation in this evaluation.

## 6. Threshold Analysis

The diagnostic script evaluates thresholds from 0.05 to 0.95 for descriptive analysis.

The best-F1 threshold found in the diagnostic sweep was **0.60**, with F1 approximately **0.0368**. This threshold is explicitly **descriptive only and was not used as a tuned or production threshold**.

At threshold 0.60:
- TP = 10
- FP = 422
- FN = 101
- TN = 16,054
- Precision ≈ 0.0231
- Recall ≈ 0.0901
- F1 ≈ 0.0368

Therefore, threshold sweeping does not establish a validated deployment operating point.

## 7. Matched-Contrast Analysis

For each positive patch, matched negatives were evaluated using the same contrast-based comparison framework.

- Positive patches: **111**
- Matching parameter k: **20**
- Win rate: **72.97%**
- Mean positive probability: **0.5276**
- Mean matched-negative probability: **0.4990**
- Median positive probability: **0.5320**
- Median matched-negative probability: **0.4904**
- Pearson correlation: **0.3842**
- Spearman correlation: **0.3609**

The contrast measure is computed from the crop+resized array before normalization.

## 8. False-Positive Analysis

At threshold 0.50:
- False positives: **7,540 / 16,476 negatives**
- False-positive rate: approximately **45.76%**

Per sample:
- `44b6_0b24845f`: 775
- `44b6_1d530831`: 4,414
- `44b6_33b596bf`: 2,351

A complete GT-in-patch contamination audit for all false positives was not computed in this diagnostic run. Consequently, the 7,540 false positives must not be described as definitively biologically false without additional exhaustive spatial auditing.

## 9. Temporal / Spatial Audit

Targeted temporal audits were performed around selected positive cases.

Audited cases:
- `44b6_33b596bf` — center frames 21, 23 and 24
- `44b6_1d530831` — center frame 92

| Sample / Case | Frames | Min distance (px) | Max distance (px) | GT inside 32 px patch |
|---|---:|---:|---:|---:|
| `33b596bf`, frame 21 | 4 | 135.36 | 150.22 | 0 |
| `1d530831`, frame 92 | 7 | 134.24 | 137.01 | 0 |
| `33b596bf`, frame 24 | 7 | 116.97 | 133.96 | 0 |
| `33b596bf`, frame 23 | 6 | 110.02 | 141.80 | 0 |

These targeted examples are far outside the **1.5 px** centering tolerance and are therefore not borderline localization mismatches.

This audit is **targeted rather than exhaustive**. It does not establish that every one of the 7,540 false positives is GT-clean.

## 10. Test Normalization Audit

Run13 computes fresh robust normalization statistics independently for each unlabeled test sample using crop+resized pixel arrays at evaluation time.

| Sample | N | Raw median | Raw MAD | Robust scale |
|---|---:|---:|---:|---:|
| `44b6_0b24845f` | 2,068 | 2191.0603 | 304.7263 | 451.7872 |
| `44b6_1d530831` | 9,763 | 2058.9834 | 509.6306 | 755.5784 |
| `44b6_33b596bf` | 4,756 | 1175.6337 | 198.5640 | 294.3909 |

Parameters:
- Robust scale constant: `1.4826`
- Epsilon floor: `1e-6`

The label column is not used when computing these statistics. This is therefore an **unlabeled sample-level transductive/unsupervised normalization procedure** and should be disclosed as part of the evaluation methodology.

This differs from the Run11 train/validation normalization, which reused Run11 saved uncropped-patch statistics.

## 11. Evidence Interpretation

Run13 provides evidence of a measurable model signal on the held-out dataset:
- ROC-AUC is above random baseline overall.
- Positive probabilities tend to exceed matched-negative probabilities in the matched-contrast analysis.
- Several test samples show positive ranking signal.

At the same time:
- Precision at threshold 0.50 is very low.
- False positives are numerous.
- Performance varies substantially across samples.
- The positive class is highly imbalanced.
- Targeted audits show examples of spatially distant detections.
- Fresh per-sample unlabeled normalization introduces a transductive evaluation characteristic.

Therefore, Run13 should be reported as a **research-stage held-out technical evaluation demonstrating measurable but insufficiently reliable detection signal**.

It should not be described as calibrated, production-ready, biologically validated, or as establishing a validated laboratory operating point.

## 12. Reproducibility Artifacts

The Run13 evidence package is supported by the following saved artifacts:

```
run13_checkpoint_sha256.txt
run13_matched_contrast_per_positive.csv
run13_per_sample_diagnostic.csv
run13_test_config.json
run13_test_diagnostic.json
run13_test_normalization_audit.json
run13_test_summary.txt
test_predictions_run13.csv
```

Primary diagnostic script:
scripts/benchmark_v2/diagnose_run13_test.py
## 13. Validation Status

| Validation item | Status |
|---|---|
| CTC benchmark methodology | Complete |
| TrackMate baseline | Complete |
| Real biological dataset | Complete |
| Benchmark-v2 distance-gated evaluation | Complete |
| Exp05 detector experiments | Complete |
| Run13 held-out evaluation | Complete |
| Original-frame visual audit | Complete |
| Temporal evidence reconciliation | Complete |
| Statistical audit | Complete |
| Normalization audit | Complete |
| Final technical validation package | In progress |
| External researcher validation | Pending |
| Real laboratory validation | Pending |

## 14. Final Technical Statement

The Run13 checkpoint establishes a reproducible held-out evaluation with documented preprocessing, normalization, dataset composition, statistical metrics, threshold analysis, and targeted spatial/temporal audits.

The evidence supports the conclusion that the current detector has measurable predictive/ranking signal, while its current operating characteristics are not sufficient to claim reliable production detection or biological validation.

Further validation should focus on exhaustive false-positive characterization, independent researcher review, external datasets, and real laboratory evaluation before deployment-oriented claims are made.
