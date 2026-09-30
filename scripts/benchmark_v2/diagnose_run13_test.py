"""
Experiment #5, Run13: held-out TEST-set evaluator.

Evaluates Run13 (stronger contrast-invariance augmentation on top of Run12 -- results/
exp05_training_dataset/run02_20samples/run13_stronger_contrast) trained by
train_exp05_run13.py, against the same frozen test split Run11/Run12 used. Does NOT modify
train_exp05_run13.py, any Run11/Run12 file, the checkpoint, manifest.csv, split_manifest.csv, or
any other existing evaluation script. Does NOT import any training utility that could mutate a
model (no optimizer, no criterion.backward(), nothing under torch.optim). Writes ONLY under a
new --out-dir.

IMPORTANT -- why this evaluator's PREPROCESSING is byte-identical to diagnose_run12_test.py's:
Run13's only intentional change vs Run12 was the raw-domain contrast-jitter scale range used
DURING TRAINING (Uniform(0.5,1.8) -> Uniform(0.25,3.0)). Contrast jitter is a training-only
augmentation -- diagnose_run12_test.py never applied it at test time either (see its own
docstring), so Run13's change has literally NO code path here. This evaluator differs from
diagnose_run12_test.py ONLY in which checkpoint/config/output paths it reads and writes -- every
preprocessing, normalization, and diagnostic function below is copied verbatim.

PREPROCESSING (test-time), exactly mirroring train_exp05_run12.py/run13.py's raw-domain step,
applied here with NO contrast jitter and NO augment_patch (both are training-only, same as
Run11/Run12's own test-time discipline):

    raw_patch (32x32, loaded from patches/test/<label>/*.npy, UNCHANGED from Stage 1)
        -> crop: raw_patch[8:24, 8:24]                              (16x16)
        -> resize: scipy.ndimage.zoom(crop, zoom=2.0, order=1)      (32x32, RESIZE_ORDER is a
                                                                       fixed module constant
                                                                       here, not a CLI flag --
                                                                       see rationale below)
        -> per-TEST-sample robust normalization, statistics computed HERE, from these
           CROP+RESIZED arrays (NOT the original 32x32 arrays -- see PROTOCOL NOTE below),
           label column never read
        -> CellCNNNoClassifierReLU -> sigmoid -> probability

RESIZE_ORDER IS HARDCODED (not exposed as a CLI argument), unlike train_exp05_run13.py's
--resize-order flag: this evaluator's whole purpose is a clean, auditable comparison against one
specific trained checkpoint, and allowing a mismatched resize order between training and
evaluation here would silently invalidate the comparison with no protection against it. The
checkpoint's own run13_config.json's recorded resize_order is read and compared against this
constant; a mismatch is a fail-closed error (see the structural check output).

CHECKPOINT SHA-256 -- NOT HARDCODED THIS TIME, unlike diagnose_run12_test.py: no known-good
checkpoint hash was provided for Run13 (it hasn't been trained yet as of writing this script).
--expected-checkpoint-sha256 is an OPTIONAL argument: if you pass it, verification is fail-closed
exactly like Run12's evaluator; if you don't, the actual hash is still computed, printed, and
recorded in every output file, and the structural check reports that row as [SKIP] rather than
silently omitting it. Once you have a trusted value (e.g. from train_exp05_run13.py's own
checkpoint_sha256.txt), pass it on every subsequent run to restore fail-closed protection.

PROTOCOL NOTE -- how this differs from Run11/Run12's TRAIN/VAL, stated once here in code and
again in run13_test_summary.txt so it can never be missed:
    TRAIN/VAL (train_exp05_run13.py): per-sample median/robust_scale were LOADED VERBATIM from
        Run11's normalization_config.json -- frozen, never recomputed -- specifically to isolate
        the contrast-jitter-range change as the only variable vs Run12.
    TEST (this script): per-TEST-sample median/robust_scale are computed FRESH, HERE, at
        evaluation time, from each test sample's own CROP+RESIZED pixels -- because Run12/Run13
        change the input representation, test statistics computed from the ORIGINAL 32x32
        arrays would not describe the distribution the model actually receives. This is the same
        category of operation Run11's own diagnose_run11_test.py and Run12's own
        diagnose_run12_test.py performed (unsupervised, per-sample, test-time-only
        normalization, label column never read) -- just recomputed on the new representation
        instead of reusing Run11's saved values, because those values describe a different
        (uncropped) pixel population. This makes TEST evaluation transductive at the sample
        level (it uses the unlabeled test sample's own pixels, available in bulk before
        per-patch classification) -- documented, not hidden.

CRITICAL LEAKAGE-SAFETY PROOF: compute_per_sample_robust_stats(X, sample_ids) below takes
EXACTLY two positional arguments -- an array of patches and an array of sample_ids. It has no
`labels` or `y` parameter anywhere in its signature. validate_no_labels_in_normalization() proves
this at runtime via inspect.signature() and prints the actual signature, rather than only
asserting it in a comment.

Does NOT modify: train_exp05_run08/09/10/11/12/13.py, diagnose_run09/11/12_test.py,
diagnose_matched_contrast.py, diagnose_cross_sample_shift.py, extract_gt_coordinates.py,
audit_gt_in_patch.py, src/model.py, src/augmentations.py, src/losses.py, src/train.py,
src/experiment.py, manifest.csv, split_manifest.csv, or the Run13 checkpoint.

Usage:
    python scripts/benchmark_v2/diagnose_run13_test.py --help

    # Structural validation only -- runs every check below, NO full inference, NO metrics,
    # NO prediction CSV written.
    python scripts/benchmark_v2/diagnose_run13_test.py `
        --run13-dir results\\exp05_training_dataset\\run02_20samples\\run13_stronger_contrast `
        --stage1-dir results\\exp05_training_dataset\\run02_20samples `
        --out-dir results\\exp05_training_dataset\\run02_20samples\\run13_test_eval `
        --structural-check-only

    # Full evaluation, after structural checks pass (add --expected-checkpoint-sha256 <hash>
    # once you have a trusted value, e.g. from train_exp05_run13.py's checkpoint_sha256.txt):
    python scripts/benchmark_v2/diagnose_run13_test.py `
        --run13-dir results\\exp05_training_dataset\\run02_20samples\\run13_stronger_contrast `
        --stage1-dir results\\exp05_training_dataset\\run02_20samples `
        --out-dir results\\exp05_training_dataset\\run02_20samples\\run13_test_eval
"""

import argparse
import csv
import hashlib
import inspect
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy
from scipy.ndimage import zoom as scipy_zoom
import sklearn
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                              accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix)

from src.model import CellCNNNoClassifierReLU  # unmodified, same class Run11/Run12/Run13 trained

LABEL_DIRS = {"positive": 1.0, "negative": 0.0}
PATCH_SIZE = 32
CROP_HALF = 8
CROP_SIZE = 16
RESIZE_ZOOM = PATCH_SIZE / CROP_SIZE       # 2.0
RESIZE_ORDER = 1                            # fixed, see module docstring
ROBUST_SCALE_CONSTANT = 1.4826
CENTERING_TOLERANCE_PX = 1.5                # matches train_exp05_run12.py/run13.py's established tolerance

EXPECTED_TEST_SAMPLES = ["44b6_0b24845f", "44b6_1d530831", "44b6_33b596bf"]
EXPECTED_N_TOTAL = 16587
EXPECTED_N_POSITIVE = 111
EXPECTED_N_NEGATIVE = 16476
EXPECTED_CHECKPOINT_SHA256 = None  # not yet known for Run13 -- see module docstring
                                     # "CHECKPOINT SHA-256" note; pass --expected-checkpoint-sha256
                                     # to enable fail-closed verification once you have one
EXPECTED_N_PARAMS = 89185

K_NEAREST_MATCHED_CONTRAST = 20
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]
DEFAULT_THRESHOLD = 0.50
TOP_N_FALSE_POSITIVES = 25


# --------------------------------------------------------------------------
# Provenance helpers -- same style as every prior Experiment script
# --------------------------------------------------------------------------

def sha256_of_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_split_manifest(path):
    assignment = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            assignment[row["sample_id"]] = row["split"]
    return assignment


def load_manifest_by_filename(manifest_csv_path):
    by_filename = {}
    with open(manifest_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            pp = row.get("patch_path", "")
            if pp:
                by_filename[Path(pp).name] = row
    return by_filename


# --------------------------------------------------------------------------
# Checkpoint discovery -- reads from the Run13 output directory, never guesses among multiple
# candidates, never touches any global/shared "best model" location.
# --------------------------------------------------------------------------

def discover_checkpoint(run13_dir, explicit_checkpoint):
    if explicit_checkpoint is not None:
        p = Path(explicit_checkpoint)
        if not p.exists():
            raise SystemExit(f"BLOCKED: --checkpoint {p} does not exist.")
        return p

    run13_dir = Path(run13_dir)
    default_path = run13_dir / "model.pth"
    if default_path.exists():
        candidates = sorted(run13_dir.glob("*.pth"))
        if len(candidates) > 1:
            raise SystemExit(
                f"BLOCKED: {run13_dir} contains {len(candidates)} .pth files "
                f"({[c.name for c in candidates]}) -- refusing to silently guess. Pass "
                f"--checkpoint explicitly."
            )
        return default_path

    all_pth = sorted(run13_dir.rglob("*.pth"))
    if len(all_pth) == 0:
        raise SystemExit(f"BLOCKED: no .pth checkpoint found anywhere under {run13_dir}.")
    if len(all_pth) > 1:
        raise SystemExit(
            f"BLOCKED: no {default_path} found, and {len(all_pth)} ambiguous .pth files exist "
            f"under {run13_dir}: {[str(c) for c in all_pth]}. Pass --checkpoint explicitly."
        )
    return all_pth[0]


# --------------------------------------------------------------------------
# Crop + resize + normalization -- duplicated from train_exp05_run12.py/run13.py verbatim (same
# no-cross-experiment-imports convention as every prior script), with RESIZE_ORDER fixed here.
# --------------------------------------------------------------------------

def crop_and_resize_patch(raw_patch, crop_half=CROP_HALF, resize_order=RESIZE_ORDER):
    if raw_patch.shape != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f"crop_and_resize_patch expects a {PATCH_SIZE}x{PATCH_SIZE} input, "
                          f"got {raw_patch.shape}.")
    center = PATCH_SIZE // 2
    crop = raw_patch[center - crop_half: center + crop_half, center - crop_half: center + crop_half]
    if crop.shape != (2 * crop_half, 2 * crop_half):
        raise ValueError(f"Center crop has shape {crop.shape}, expected "
                          f"({2*crop_half},{2*crop_half}).")
    zoom_factor = PATCH_SIZE / (2 * crop_half)
    resized = scipy_zoom(crop.astype(np.float64), zoom=zoom_factor, order=resize_order)
    if resized.shape != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError(f"Resized patch has shape {resized.shape}, expected "
                          f"({PATCH_SIZE},{PATCH_SIZE}).")
    return resized.astype(np.float32)


def compute_per_sample_robust_stats(X, sample_ids, min_patches_warn=200):
    """LEAKAGE-SAFETY: exactly two parameters, X and sample_ids -- no labels parameter exists.
    Computed here on CROP+RESIZED arrays (the caller passes X_crop_resized, never
    X_original_32x32) -- see PROTOCOL NOTE in the module docstring."""
    stats = {}
    for sid in sorted(set(sample_ids.tolist())):
        mask = sample_ids == sid
        n_patches = int(mask.sum())
        if n_patches < min_patches_warn:
            print(f"  WARNING: sample '{sid}' has only {n_patches} patches (< {min_patches_warn}).")
        pixels = X[mask].astype(np.float64).ravel()
        median = float(np.median(pixels))
        mad = float(np.median(np.abs(pixels - median)))
        stats[sid] = {"n_patches": n_patches, "raw_median": median, "raw_mad": mad,
                      "robust_scale": ROBUST_SCALE_CONSTANT * mad}
    return stats


def normalize_patch_sample_robust(patch, sample_median, sample_scale, eps):
    patch64 = patch.astype(np.float64)
    return ((patch64 - sample_median) / (sample_scale + eps)).astype(np.float32)


def validate_no_labels_in_normalization():
    sig = inspect.signature(compute_per_sample_robust_stats)
    param_names = list(sig.parameters.keys())
    has_label_param = any("label" in p.lower() or p.lower() == "y" for p in param_names)
    return (not has_label_param), str(sig), param_names


# --------------------------------------------------------------------------
# Scan / load
# --------------------------------------------------------------------------

def scan_and_validate_test_patches(patches_dir, manifest_by_filename, test_sample_ids):
    test_dir = Path(patches_dir) / "test"
    all_patches = []
    counts_by_label = {"positive": 0, "negative": 0}
    for label_name, label_value in LABEL_DIRS.items():
        label_dir = test_dir / label_name
        if not label_dir.is_dir():
            raise FileNotFoundError(f"BLOCKED: required directory missing: {label_dir}")
        files = sorted(label_dir.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"BLOCKED: no .npy files found in {label_dir}")
        for f in files:
            row = manifest_by_filename.get(f.name)
            if row is None:
                raise ValueError(f"BLOCKED: {f} has no matching manifest.csv row (by filename).")
            if row["label"] != label_name:
                raise ValueError(f"BLOCKED: {f} label mismatch vs manifest.csv ('{row['label']}').")
            if row["split"] != "test":
                raise ValueError(f"BLOCKED: {f} split mismatch vs manifest.csv ('{row['split']}').")
            if row["sample_id"] not in test_sample_ids:
                raise ValueError(f"BLOCKED: {f} sample_id '{row['sample_id']}' not in the frozen test set.")
            all_patches.append({"path": f, "label": label_value, "sample_id": row["sample_id"],
                                 "filename": f.name, "frame_idx": row.get("frame_idx"),
                                 "candidate_y": row.get("candidate_y"), "candidate_x": row.get("candidate_x")})
            counts_by_label[label_name] += 1
    return all_patches, counts_by_label


def load_all_raw_test_patches(all_patches):
    X = np.empty((len(all_patches), PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    sample_ids = np.empty(len(all_patches), dtype=object)
    for i, p in enumerate(all_patches):
        arr = np.load(p["path"])
        if arr.shape != (PATCH_SIZE, PATCH_SIZE):
            raise ValueError(f"SHAPE MISMATCH: {p['path']} has shape {arr.shape}.")
        if arr.dtype != np.float32:
            raise ValueError(f"DTYPE MISMATCH: {p['path']} has dtype {arr.dtype}.")
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError(f"NaN/Inf VALUES FOUND in {p['path']}.")
        X[i] = arr
        sample_ids[i] = p["sample_id"]
    return X, sample_ids


# --------------------------------------------------------------------------
# Structural validation -- every check the task requires, each an explicit PASS/FAIL line.
# Uses only a probe subset for the expensive per-patch checks (shape/centering/NaN/Inf), and the
# full scan for count checks. Does NOT run the model over the full 16,587 patches.
# --------------------------------------------------------------------------

def run_structural_checks(patches_dir, manifest_csv_path, split_manifest_path, checkpoint_path,
                           run13_config, expected_checkpoint_sha256, probe_size=300, seed=42):
    print("\n" + "=" * 70)
    print("STRUCTURAL VALIDATION")
    print("=" * 70)
    results = []

    def check(name, passed, detail="", skip=False):
        status = "SKIP" if skip else ("PASS" if passed else "FAIL")
        print(f"[{status}] {name}{' -- ' + detail if detail else ''}")
        if not skip:
            results.append((name, passed))
        return passed

    split_assignment = load_split_manifest(split_manifest_path)
    test_ids = sorted(sid for sid, sp in split_assignment.items() if sp == "test")
    check("test split IDs exactly match Run11's frozen test IDs",
          test_ids == sorted(EXPECTED_TEST_SAMPLES),
          f"found {test_ids}, expected {sorted(EXPECTED_TEST_SAMPLES)}")

    manifest_by_filename = load_manifest_by_filename(manifest_csv_path)
    all_patches, counts_by_label = scan_and_validate_test_patches(
        patches_dir, manifest_by_filename, set(test_ids))
    n_total = len(all_patches)
    check("exactly 16,587 test patches", n_total == EXPECTED_N_TOTAL,
          f"found {n_total}")
    check("exactly 111 positive / 16,476 negative (labels read AFTER this structural scan, "
          "not used for normalization anywhere in this script)",
          counts_by_label["positive"] == EXPECTED_N_POSITIVE and
          counts_by_label["negative"] == EXPECTED_N_NEGATIVE,
          f"found positive={counts_by_label['positive']} negative={counts_by_label['negative']}")

    rng = np.random.default_rng(seed)
    probe_idx = rng.choice(len(all_patches), size=min(probe_size, len(all_patches)), replace=False)
    shape_ok = True
    crop_shape_ok = True
    resize_shape_ok = True
    nan_inf_ok = True
    for i in probe_idx:
        arr = np.load(all_patches[i]["path"])
        if arr.shape != (32, 32):
            shape_ok = False
        center = PATCH_SIZE // 2
        crop = arr[center - CROP_HALF: center + CROP_HALF, center - CROP_HALF: center + CROP_HALF]
        if crop.shape != (16, 16):
            crop_shape_ok = False
        resized = crop_and_resize_patch(arr)
        if resized.shape != (32, 32):
            resize_shape_ok = False
        if np.isnan(resized).any() or np.isinf(resized).any():
            nan_inf_ok = False
    check(f"all source patches are 32x32 (probe n={len(probe_idx)})", shape_ok)
    check("crop is 16x16", crop_shape_ok)
    check("resized patches are 32x32", resize_shape_ok)
    check("no NaN/Inf after crop+resize (probe)", nan_inf_ok)

    yy, xx = np.mgrid[0:16, 0:16]
    synthetic_crop = np.exp(-(((yy - CROP_HALF) ** 2 + (xx - CROP_HALF) ** 2) / 8.0))
    synthetic_resized = scipy_zoom(synthetic_crop, zoom=RESIZE_ZOOM, order=RESIZE_ORDER)
    ys, xs = np.mgrid[0:synthetic_resized.shape[0], 0:synthetic_resized.shape[1]]
    cy = (ys * synthetic_resized).sum() / synthetic_resized.sum()
    cx = (xs * synthetic_resized).sum() / synthetic_resized.sum()
    offset = float(np.hypot(cy - 16, cx - 16))
    check(f"candidate remains centered within {CENTERING_TOLERANCE_PX}px tolerance",
          offset <= CENTERING_TOLERANCE_PX,
          f"measured offset={offset:.3f}px (established value from train_exp05_run12.py/run13.py)")

    model = CellCNNNoClassifierReLU()
    n_params = sum(p.numel() for p in model.parameters())
    check(f"model has exactly {EXPECTED_N_PARAMS} parameters", n_params == EXPECTED_N_PARAMS,
          f"found {n_params}")

    actual_sha256 = sha256_of_file(checkpoint_path)
    if expected_checkpoint_sha256 is None:
        check("checkpoint SHA-256 verification", None, skip=True,
              detail=f"no --expected-checkpoint-sha256 given -- actual hash: {actual_sha256} "
                     f"(recorded in every output file regardless; pass this value on a future "
                     f"run to enable fail-closed verification)")
    else:
        check("checkpoint SHA-256 matches expected value", actual_sha256 == expected_checkpoint_sha256,
              f"expected {expected_checkpoint_sha256}\n         got      {actual_sha256}")

    no_labels, sig_str, param_names = validate_no_labels_in_normalization()
    check("normalization-stat computation does not receive labels", no_labels,
          f"compute_per_sample_robust_stats{sig_str}, parameters={param_names}")

    check("test normalization statistics are computed from crop+resized arrays, not original "
          "32x32 arrays (structural: compute_test_normalization_stats() below only ever calls "
          "compute_per_sample_robust_stats() on X_crop_resized, never on X_raw -- verified by "
          "code path, not a runtime probe)", True,
          "see compute_test_normalization_stats() -- X_raw is never passed to "
          "compute_per_sample_robust_stats() anywhere in this script")

    if run13_config is not None:
        recorded_resize_order = run13_config.get("field_of_view_change", {}).get("resize_order")
        check("this evaluator's fixed RESIZE_ORDER matches Run13 training's recorded resize_order",
              recorded_resize_order == RESIZE_ORDER,
              f"training recorded resize_order={recorded_resize_order}, evaluator uses "
              f"RESIZE_ORDER={RESIZE_ORDER}")

    n_fail = sum(1 for _, ok in results if not ok)
    print("=" * 70)
    print(f"STRUCTURAL VALIDATION COMPLETE: {len(results)-n_fail}/{len(results)} passed"
          + (f", {n_fail} FAILED" if n_fail else ""))
    print("=" * 70)
    return results, all_patches, counts_by_label, test_ids


# --------------------------------------------------------------------------
# Diagnostics: matched-contrast, cross-sample, false-positive analysis. All computed here,
# self-contained -- do not shell out to diagnose_matched_contrast.py / diagnose_cross_sample_
# shift.py, since those compute their contrast statistic from the ORIGINAL 32x32 patch, not the
# crop-resized representation Run13's model actually receives.
# --------------------------------------------------------------------------

def matched_contrast_diagnostic(records, k=K_NEAREST_MATCHED_CONTRAST):
    """records: list of dicts with sample_id, label, probability, contrast_std (std of the
    CROP-RESIZED array, pre-normalization). Matching is deterministic (nearest by absolute
    |std_i - std_j|, ties broken by original list order via a stable sort) -- no RNG needed."""
    by_sample = {}
    for r in records:
        by_sample.setdefault(r["sample_id"], {"pos": [], "neg": []})
        by_sample[r["sample_id"]]["pos" if r["label"] == 1.0 else "neg"].append(r)

    per_positive_rows = []
    for sid, groups in by_sample.items():
        negs = groups["neg"]
        neg_stds = np.array([n["contrast_std"] for n in negs])
        for pos in groups["pos"]:
            diffs = np.abs(neg_stds - pos["contrast_std"])
            order = np.argsort(diffs, kind="stable")
            k_actual = min(k, len(order))
            matched = [negs[i] for i in order[:k_actual]]
            matched_probs = np.array([m["probability"] for m in matched])
            per_positive_rows.append({
                "sample_id": sid, "filename": pos["filename"], "positive_prob": pos["probability"],
                "matched_mean_neg_prob": float(matched_probs.mean()),
                "matched_median_neg_prob": float(np.median(matched_probs)),
                "beats_matched_mean": bool(pos["probability"] > matched_probs.mean()),
                "k_used": k_actual,
            })

    if not per_positive_rows:
        return None, per_positive_rows

    win_rate = 100.0 * np.mean([r["beats_matched_mean"] for r in per_positive_rows])
    mean_pos = np.mean([r["positive_prob"] for r in per_positive_rows])
    mean_matched_neg = np.mean([r["matched_mean_neg_prob"] for r in per_positive_rows])
    median_pos = np.median([r["positive_prob"] for r in per_positive_rows])
    median_matched_neg = np.median([r["matched_median_neg_prob"] for r in per_positive_rows])

    all_probs = np.array([r["probability"] for r in records])
    all_stds = np.array([r["contrast_std"] for r in records])
    pearson_r = float(np.corrcoef(all_probs, all_stds)[0, 1])
    from scipy.stats import spearmanr
    spearman_r = float(spearmanr(all_probs, all_stds).statistic)

    summary = {
        "k": k, "n_positives": len(per_positive_rows),
        "win_rate_pct": float(win_rate),
        "mean_positive_prob": float(mean_pos), "mean_matched_negative_prob": float(mean_matched_neg),
        "median_positive_prob": float(median_pos), "median_matched_negative_prob": float(median_matched_neg),
        "pearson_r_probability_vs_crop_resized_contrast_std": pearson_r,
        "spearman_r_probability_vs_crop_resized_contrast_std": spearman_r,
        "contrast_statistic_source": "std of the CROP+RESIZED array, pre-normalization (NOT "
            "the original 32x32 patch's std) -- consistent with the field-of-view Run13's model "
            "actually sees.",
    }
    return summary, per_positive_rows


def cross_sample_diagnostic(records, test_ids):
    out = {}
    for sid in test_ids:
        sub = [r for r in records if r["sample_id"] == sid]
        yt = np.array([r["label"] for r in sub])
        pr = np.array([r["probability"] for r in sub])
        n_pos = int((yt == 1.0).sum())
        n_neg = int((yt == 0.0).sum())
        auc = float(roc_auc_score(yt, pr)) if 0 < n_pos < len(yt) else float("nan")
        pr_auc = float(average_precision_score(yt, pr)) if n_pos > 0 else float("nan")
        out[sid] = {
            "n": len(sub), "n_positive": n_pos, "n_negative": n_neg,
            "roc_auc": auc, "pr_auc": pr_auc,
            "positive_mean_probability": float(pr[yt == 1.0].mean()) if n_pos else float("nan"),
            "negative_mean_probability": float(pr[yt == 0.0].mean()) if n_neg else float("nan"),
        }
    return out


def false_positive_analysis(records, threshold=DEFAULT_THRESHOLD, top_n=TOP_N_FALSE_POSITIVES):
    negs = [r for r in records if r["label"] == 0.0]
    fps = [r for r in negs if r["probability"] >= threshold]
    by_sample = {}
    for r in fps:
        by_sample[r["sample_id"]] = by_sample.get(r["sample_id"], 0) + 1
    top = sorted(negs, key=lambda r: -r["probability"])[:top_n]
    return {
        "threshold": threshold,
        "total_false_positives": len(fps),
        "n_negatives": len(negs),
        "false_positive_rate_pct": 100.0 * len(fps) / len(negs) if negs else float("nan"),
        "false_positives_by_sample": by_sample,
        "top_negatives": [{"sample_id": r["sample_id"], "filename": r["filename"],
                           "probability": r["probability"]} for r in top],
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Held-out TEST-set evaluator for Experiment #5 Run13 (stronger "
                     "contrast-invariance augmentation on top of Run12). Structural-checks-only "
                     "mode available via --structural-check-only; does not run inference or "
                     "write predictions."
    )
    parser.add_argument("--run13-dir", required=True,
                         help="Run13 training output directory, e.g. "
                              "results/exp05_training_dataset/run02_20samples/run13_stronger_contrast")
    parser.add_argument("--stage1-dir", required=True,
                         help="e.g. results/exp05_training_dataset/run02_20samples")
    parser.add_argument("--patches-dir", default=None, help="Default: <stage1-dir>/patches")
    parser.add_argument("--manifest-csv", default=None, help="Default: <stage1-dir>/manifest.csv")
    parser.add_argument("--split-manifest", default=None, help="Default: <stage1-dir>/split_manifest.csv")
    parser.add_argument("--checkpoint", default=None,
                         help="Explicit checkpoint path. Default: auto-discover a single "
                              "unambiguous model.pth under --run13-dir (fails closed if none or "
                              "more than one is found).")
    parser.add_argument("--expected-checkpoint-sha256", default=None,
                         help="Optional known-good checkpoint SHA-256 (e.g. from "
                              "train_exp05_run13.py's own checkpoint_sha256.txt). If given, "
                              "verification is fail-closed. If omitted, the actual hash is still "
                              "computed and recorded everywhere, but the check is reported as "
                              "[SKIP] rather than compared against nothing.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-nonempty-out-dir", action="store_true")
    parser.add_argument("--structural-check-only", action="store_true",
                         help="Run every structural/leakage validation check and stop. No "
                              "inference over the full test set, no metrics, no prediction CSV.")
    parser.add_argument("--gt-audit-csv", default=None,
                         help="Optional path to audit_gt_in_patch_per_candidate.csv, for an "
                              "OPTIONAL false-positive contamination cross-reference. Evaluation "
                              "runs identically with or without this.")
    args = parser.parse_args()

    run13_dir = Path(args.run13_dir)
    stage1_dir = Path(args.stage1_dir)
    patches_dir = Path(args.patches_dir) if args.patches_dir else stage1_dir / "patches"
    manifest_csv_path = Path(args.manifest_csv) if args.manifest_csv else stage1_dir / "manifest.csv"
    split_manifest_path = Path(args.split_manifest) if args.split_manifest else stage1_dir / "split_manifest.csv"
    out_dir = Path(args.out_dir)

    for p in [run13_dir, stage1_dir, patches_dir, manifest_csv_path, split_manifest_path]:
        if not p.exists():
            raise SystemExit(f"BLOCKED: required path does not exist: {p}")

    run13_config_path = run13_dir / "run13_config.json"
    run13_config = json.loads(run13_config_path.read_text()) if run13_config_path.exists() else None
    if run13_config is None:
        print(f"NOTE: {run13_config_path} not found -- skipping the resize_order "
              f"cross-check against training config (all other checks still run).")

    checkpoint_path = discover_checkpoint(run13_dir, args.checkpoint)
    print(f"Checkpoint: {checkpoint_path}")

    run_timestamp_utc = datetime.now(timezone.utc).isoformat()
    print(f"Experiment #5 Run13 test evaluation -- "
          f"{'STRUCTURAL CHECK ONLY' if args.structural_check_only else 'FULL EVALUATION'}")
    print(f"Timestamp (UTC): {run_timestamp_utc}")
    print(f"Command line: {' '.join(sys.argv)}\n")

    results, all_patches, counts_by_label, test_ids = run_structural_checks(
        patches_dir, manifest_csv_path, split_manifest_path, checkpoint_path, run13_config,
        args.expected_checkpoint_sha256, seed=args.seed,
    )
    n_fail = sum(1 for _, ok in results if not ok)
    if n_fail:
        raise SystemExit(f"BLOCKED: {n_fail} structural check(s) failed. Fix before evaluating.")

    if args.structural_check_only:
        print("\n--structural-check-only: all checks passed. NOTHING was written under "
              "--out-dir. No inference was run. Re-run without this flag for the full "
              "evaluation.")
        return

    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_nonempty_out_dir:
        raise SystemExit(f"Refusing to write into non-empty existing directory {out_dir} "
                          f"(pass --allow-nonempty-out-dir to override).")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading all test patches into memory...")
    X_raw, sample_ids = load_all_raw_test_patches(all_patches)

    print("Applying crop+resize to every test patch (RESIZE_ORDER={})...".format(RESIZE_ORDER))
    X_crop_resized = np.stack([crop_and_resize_patch(X_raw[i]) for i in range(len(X_raw))])

    print("\nComputing per-TEST-sample robust normalization statistics from CROP+RESIZED "
          "pixels (unsupervised, test-time-only, label column never read)...")
    stats_test = compute_per_sample_robust_stats(X_crop_resized, sample_ids)
    for sid, s in sorted(stats_test.items()):
        print(f"    {sid}: n_patches={s['n_patches']} median={s['raw_median']:.2f} "
              f"MAD={s['raw_mad']:.2f} robust_scale={s['robust_scale']:.2f}")

    eps = 1e-6  # matches train_exp05_run11.py/run12.py/run13.py's NORM_EPS_FLOOR; per-sample MAD
                # over thousands of crop-resized pixels cannot plausibly approach zero
    test_norm_audit_path = out_dir / "run13_test_normalization_audit.json"
    test_norm_audit_path.write_text(json.dumps({
        "run_timestamp_utc": run_timestamp_utc,
        "note": "Computed from crop+resized TEST pixels at evaluation time. Label column never "
                "read. See module docstring PROTOCOL NOTE for why this differs from Run11 "
                "train/val (which reused Run11's own SAVED, uncropped-patch statistics).",
        "eps_floor": eps, "robust_scale_constant": ROBUST_SCALE_CONSTANT,
        "per_sample_stats": stats_test,
    }, indent=2))
    print(f"Saved: {test_norm_audit_path}")

    device = torch.device(args.device)
    model = CellCNNNoClassifierReLU().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"\nCheckpoint loaded: epoch={ckpt.get('epoch')} val_pr_auc={ckpt.get('val_pr_auc')}")

    print("\nRunning inference...")
    records = []
    n = len(all_patches)
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            batch_arrs = []
            batch_stds = []
            for i in range(start, end):
                sid = sample_ids[i]
                s = stats_test[sid]
                cr = X_crop_resized[i]
                batch_stds.append(float(cr.std()))
                normed = normalize_patch_sample_robust(cr, s["raw_median"], s["robust_scale"], eps)
                batch_arrs.append(normed)
            batch_arr = np.stack(batch_arrs, axis=0)[:, None, :, :]
            images = torch.from_numpy(batch_arr).to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            for j, prob, cstd in zip(range(start, end), probs, batch_stds):
                p = all_patches[j]
                records.append({"sample_id": p["sample_id"], "label": p["label"],
                                "probability": float(prob), "filename": p["filename"],
                                "patch_path": str(p["path"]), "contrast_std": cstd})
            if end % 2000 < args.batch_size or end == n:
                print(f"  inference: {end} / {n}")

    predictions_csv_path = out_dir / "test_predictions_run13.csv"
    with open(predictions_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "label", "probability", "filename", "patch_path"])
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in ["sample_id", "label", "probability", "filename", "patch_path"]})
    print(f"Saved: {predictions_csv_path}")

    y_true = np.array([r["label"] for r in records], dtype=np.float32)
    probs = np.array([r["probability"] for r in records], dtype=np.float64)
    preds_50 = (probs >= DEFAULT_THRESHOLD).astype(int)

    roc_auc = float(roc_auc_score(y_true, probs))
    pr_auc = float(average_precision_score(y_true, probs))
    balanced_acc = float(balanced_accuracy_score(y_true, preds_50))
    acc = float(accuracy_score(y_true, preds_50))
    prec = float(precision_score(y_true, preds_50, zero_division=0))
    rec = float(recall_score(y_true, preds_50, zero_division=0))
    f1 = float(f1_score(y_true, preds_50, zero_division=0))
    cm = confusion_matrix(y_true, preds_50).tolist()

    def confusion_at(t):
        pr = (probs >= t).astype(int)
        tp = int(np.sum((pr == 1) & (y_true == 1))); fp = int(np.sum((pr == 1) & (y_true == 0)))
        tn = int(np.sum((pr == 0) & (y_true == 0))); fn = int(np.sum((pr == 0) & (y_true == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1_ = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 and not (np.isnan(precision) or np.isnan(recall)) else float("nan"))
        return {"threshold": t, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision": precision, "recall": recall, "f1": f1_}

    threshold_table = [confusion_at(t) for t in THRESHOLDS]
    best_f1_row = max(threshold_table, key=lambda r: (r["f1"] if not np.isnan(r["f1"]) else -1))

    pos_probs = probs[y_true == 1.0]
    neg_probs = probs[y_true == 0.0]
    prob_stats = {
        "positive_prevalence_pct": 100.0 * len(pos_probs) / len(probs),
        "positive_mean_probability": float(pos_probs.mean()), "positive_median_probability": float(np.median(pos_probs)),
        "negative_mean_probability": float(neg_probs.mean()), "negative_median_probability": float(np.median(neg_probs)),
        "probability_percentiles_overall": {str(q): float(np.percentile(probs, q)) for q in [5, 25, 50, 75, 90, 95]},
    }

    matched_summary, matched_rows = matched_contrast_diagnostic(records)
    cross_sample = cross_sample_diagnostic(records, test_ids)
    fp_analysis = false_positive_analysis(records)

    gt_contamination_note = "Not computed (--gt-audit-csv not provided or not found)."
    if args.gt_audit_csv and Path(args.gt_audit_csv).exists():
        import collections
        gt_rows = {}
        with open(args.gt_audit_csv, newline="") as f:
            for row in csv.DictReader(f):
                key = (row["sample_id"], row.get("frame_idx"), row.get("candidate_y"), row.get("candidate_x"))
                gt_rows[key] = row
        n_with_key = 0
        n_contaminated = 0
        for p in all_patches:
            key = (p["sample_id"], str(p.get("frame_idx")), str(p.get("candidate_y")), str(p.get("candidate_x")))
            if key in gt_rows:
                n_with_key += 1
                if int(float(gt_rows[key].get("n_gt_inside_patch", 0))) > 0:
                    n_contaminated += 1
        gt_contamination_note = (f"Joined {n_with_key}/{len(all_patches)} test rows against "
                                 f"{args.gt_audit_csv}; {n_contaminated} had >=1 GT inside their "
                                 f"patch (optional cross-reference only, not used for metrics).")
    print(f"\nGT-in-patch contamination note: {gt_contamination_note}")

    # ---- config / provenance ----
    config_record = {
        "run_timestamp_utc": run_timestamp_utc, "command_line": " ".join(sys.argv),
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": sha256_of_file(checkpoint_path),
        "split_manifest_sha256": sha256_of_file(split_manifest_path),
        "script_sha256": sha256_of_file(Path(__file__)),
        "model_class": "CellCNNNoClassifierReLU",
        "model_n_parameters": sum(p.numel() for p in CellCNNNoClassifierReLU().parameters()),
        "crop_coordinates": f"patch[{PATCH_SIZE//2-CROP_HALF}:{PATCH_SIZE//2+CROP_HALF}, "
                            f"{PATCH_SIZE//2-CROP_HALF}:{PATCH_SIZE//2+CROP_HALF}]",
        "resize_method": "scipy.ndimage.zoom", "resize_order": RESIZE_ORDER, "resize_zoom": RESIZE_ZOOM,
        "normalization_method": "per-TEST-sample robust (median/MAD), computed from crop+resized "
                                "pixels at evaluation time -- see PROTOCOL NOTE",
        "test_sample_ids": test_ids, "seed": args.seed, "device": str(device),
        "library_versions": {"torch": torch.__version__, "numpy": np.__version__,
                             "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                             "python": platform.python_version()},
    }
    (out_dir / "run13_test_config.json").write_text(json.dumps(config_record, indent=2))
    (out_dir / "run13_checkpoint_sha256.txt").write_text(config_record["checkpoint_sha256"] + "\n")

    diagnostic = {
        "n_patches": len(records), "n_positive": int((y_true == 1.0).sum()), "n_negative": int((y_true == 0.0).sum()),
        "roc_auc": roc_auc, "pr_auc": pr_auc, "balanced_accuracy": balanced_acc, "accuracy": acc,
        "precision": prec, "recall": rec, "f1": f1, "confusion_matrix_at_0.50": cm,
        "best_f1_threshold_DESCRIPTIVE_ONLY_NOT_TUNED": best_f1_row,
        "probability_stats": prob_stats,
        "cross_sample": cross_sample,
        "matched_contrast": matched_summary,
        "false_positive_analysis": fp_analysis,
    }
    (out_dir / "run13_test_diagnostic.json").write_text(json.dumps(diagnostic, indent=2))

    with open(out_dir / "run13_matched_contrast_per_positive.csv", "w", newline="") as f:
        if matched_rows:
            w = csv.DictWriter(f, fieldnames=list(matched_rows[0].keys()))
            w.writeheader(); w.writerows(matched_rows)

    with open(out_dir / "run13_per_sample_diagnostic.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "n", "n_positive", "n_negative", "roc_auc",
                                          "pr_auc", "positive_mean_probability", "negative_mean_probability"])
        w.writeheader()
        for sid, d in cross_sample.items():
            w.writerow({"sample_id": sid, **d})

    summary_lines = [
        "RUN13 TEST DIAGNOSTIC", "=" * 78,
        "IMPORTANT PROTOCOL DISTINCTION -- READ FIRST",
        "  TRAIN/VAL (train_exp05_run13.py): Run11's per-sample normalization statistics were",
        "  LOADED and FROZEN -- never recomputed -- specifically to isolate Run13's intentional",
        "  change (wider contrast-jitter range) as the only variable vs Run12. The field-of-view",
        "  change (center-crop-16) is inherited unchanged from Run12 and is also NOT recomputed.",
        "  TEST (this script): fresh, UNLABELED per-sample robust statistics are computed HERE,",
        "  at evaluation time, from each test sample's own CROP+RESIZED pixels -- because Run12/",
        "  Run13 change the input representation, so Run11's saved (uncropped) statistics would",
        "  not describe what the model actually receives. This makes TEST evaluation transductive",
        "  at the sample level (unsupervised use of each unlabeled test sample's own pixel",
        "  distribution) -- stated plainly here, not hidden.",
        "=" * 78,
        f"n={len(records)} positive={int((y_true==1.0).sum())} negative={int((y_true==0.0).sum())}",
        f"ROC-AUC={roc_auc:.6f}  PR-AUC={pr_auc:.6f}  BalancedAcc={balanced_acc:.4f}  "
        f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}",
        f"Confusion matrix @0.50: {cm}",
        f"Best-F1 threshold (DESCRIPTIVE ONLY -- NOT used as a tuned/production threshold): "
        f"{best_f1_row['threshold']} (F1={best_f1_row['f1']:.4f})",
        "",
        "--- Per-sample ---",
    ]
    for sid, d in cross_sample.items():
        summary_lines.append(f"  {sid}: n={d['n']} pos={d['n_positive']} neg={d['n_negative']} "
                             f"ROC-AUC={d['roc_auc']:.6f} PR-AUC={d['pr_auc']:.6f} "
                             f"pos_mean_prob={d['positive_mean_probability']:.4f} "
                             f"neg_mean_prob={d['negative_mean_probability']:.4f}")
    if matched_summary:
        summary_lines += ["", "--- Matched-contrast diagnostic ---",
                          f"  k={matched_summary['k']}  n_positives={matched_summary['n_positives']}",
                          f"  win_rate={matched_summary['win_rate_pct']:.2f}%",
                          f"  mean positive prob={matched_summary['mean_positive_prob']:.4f} vs "
                          f"mean matched-negative prob={matched_summary['mean_matched_negative_prob']:.4f}",
                          f"  median positive prob={matched_summary['median_positive_prob']:.4f} vs "
                          f"median matched-negative prob={matched_summary['median_matched_negative_prob']:.4f}",
                          f"  Pearson r={matched_summary['pearson_r_probability_vs_crop_resized_contrast_std']:.4f}  "
                          f"Spearman r={matched_summary['spearman_r_probability_vs_crop_resized_contrast_std']:.4f}",
                          f"  contrast statistic source: {matched_summary['contrast_statistic_source']}"]
    summary_lines += ["", "--- False-positive analysis (threshold=0.50) ---",
                      f"  total FP={fp_analysis['total_false_positives']} / {fp_analysis['n_negatives']} negatives "
                      f"({fp_analysis['false_positive_rate_pct']:.2f}%)",
                      f"  by sample: {fp_analysis['false_positives_by_sample']}",
                      f"  GT-in-patch contamination cross-reference: {gt_contamination_note}"]
    (out_dir / "run13_test_summary.txt").write_text("\n".join(summary_lines))

    print("\n" + "\n".join(summary_lines))
    print(f"\nSaved: {out_dir / 'run13_test_summary.txt'}")
    print(f"Saved: {out_dir / 'run13_test_diagnostic.json'}")
    print(f"Saved: {out_dir / 'run13_test_config.json'}")
    print(f"Saved: {out_dir / 'run13_checkpoint_sha256.txt'}")


if __name__ == "__main__":
    main()
