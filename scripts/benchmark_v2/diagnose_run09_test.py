"""
Experiment #5, Run09 diagnostic: final held-out TEST-set evaluation.

Mirrors diagnose_run08_test.py's structure and sanity-check discipline exactly (same expected
counts, same Tier-1 on-disk scan, same resumable-predictions pattern) so Run08 and Run09 test
predictions are produced under an identical evaluation harness. The ONE deliberate difference:
normalization uses Run09's own robust_eps (loaded from THIS run's normalization_config.json,
written by train_exp05_run09.py from TRAINING data only) instead of Run08's fixed 1e-6 -- that
difference is part of the Run09 METHOD being evaluated, not a change in evaluation methodology.

Does NOT modify: train_exp05_run08.py, train_exp05_run09.py, diagnose_run08_test.py,
diagnose_cross_sample_shift.py, manifest.csv, split_manifest.csv, or the Run09 checkpoint.

Usage:
    python scripts\\benchmark_v2\\diagnose_run09_test.py --overwrite
    python scripts\\benchmark_v2\\diagnose_run09_test.py --reuse-predictions
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score,
                              accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix)

from src.model import CellCNN  # unmodified

TEST_SAMPLES_DEFAULT = None  # loaded from split_manifest.csv, never hardcoded/recomputed here
LABEL_DIRS = {"positive": 1.0, "negative": 0.0}
PATCH_SIZE = 32
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]
DEFAULT_THRESHOLD = 0.50


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
            all_patches.append({"path": f, "label": label_value, "sample_id": row["sample_id"], "filename": f.name})
            counts_by_label[label_name] += 1
    return all_patches, counts_by_label


def normalize_patch_instance(arr, eps):
    arr64 = arr.astype(np.float64)
    mean = arr64.mean()
    std = arr64.std()
    return ((arr64 - mean) / (std + eps)).astype(np.float32)


def run_inference(all_patches, model, device, batch_size, eps, progress_every):
    predictions = []
    n = len(all_patches)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = all_patches[start:start + batch_size]
            arrs = []
            for p in batch:
                arr = np.load(p["path"])
                if arr.shape != (PATCH_SIZE, PATCH_SIZE):
                    raise ValueError(f"SHAPE MISMATCH: {p['path']} has shape {arr.shape}.")
                if arr.dtype != np.float32:
                    raise ValueError(f"DTYPE MISMATCH: {p['path']} has dtype {arr.dtype}.")
                if np.isnan(arr).any() or np.isinf(arr).any():
                    raise ValueError(f"NaN/Inf VALUES FOUND in {p['path']}.")
                arrs.append(normalize_patch_instance(arr, eps))
            batch_arr = np.stack(arrs, axis=0)[:, None, :, :]
            images = torch.from_numpy(batch_arr).to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            for p, prob in zip(batch, probs):
                predictions.append({"sample_id": p["sample_id"], "label": p["label"],
                                     "probability": float(prob), "filename": p["filename"],
                                     "patch_path": str(p["path"])})
            done = min(start + batch_size, n)
            if progress_every and (done % progress_every < batch_size or done == n):
                print(f"inference: {done} / {n}")
    return predictions


def compute_confusion_at_threshold(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 and not (np.isnan(precision) or np.isnan(recall)) else float("nan"))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return {"threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "specificity": specificity}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="e.g. results/.../run09_contrast_aug_focal_robust_eps")
    parser.add_argument("--stage1-root", required=True, help="e.g. results/exp05_training_dataset/run02_20samples")
    parser.add_argument("--checkpoint", default=None, help="Default: <run-dir>/model.pth")
    parser.add_argument("--patches-dir", default=None, help="Default: <stage1-root>/patches")
    parser.add_argument("--manifest-csv", default=None, help="Default: <stage1-root>/manifest.csv")
    parser.add_argument("--split-manifest", default=None, help="Default: <stage1-root>/split_manifest.csv")
    parser.add_argument("--normalization-config", default=None, help="Default: <run-dir>/normalization_config.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress-every", type=int, default=2000)
    parser.add_argument("--reuse-predictions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    stage1_root = Path(args.stage1_root)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "model.pth"
    patches_dir = Path(args.patches_dir) if args.patches_dir else stage1_root / "patches"
    manifest_csv_path = Path(args.manifest_csv) if args.manifest_csv else stage1_root / "manifest.csv"
    split_manifest_path = Path(args.split_manifest) if args.split_manifest else stage1_root / "split_manifest.csv"
    normalization_config_path = Path(args.normalization_config) if args.normalization_config else run_dir / "normalization_config.json"

    predictions_csv_path = run_dir / "test_predictions_resumable.csv"
    diagnostic_json_path = run_dir / "run09_test_diagnostic.json"
    diagnostic_txt_path = run_dir / "run09_test_diagnostic.txt"

    if not args.overwrite and not args.reuse_predictions:
        for p in [diagnostic_json_path, diagnostic_txt_path]:
            if p.exists():
                raise SystemExit(f"BLOCKED: {p} already exists. Pass --overwrite or --reuse-predictions.")

    split_assignment = load_split_manifest(split_manifest_path)
    test_sample_ids = {sid for sid, sp in split_assignment.items() if sp == "test"}
    print(f"Frozen test samples (loaded from split_manifest.csv, never recomputed): {sorted(test_sample_ids)}")

    if not normalization_config_path.exists():
        raise SystemExit(f"BLOCKED: {normalization_config_path} not found -- run train_exp05_run09.py "
                          f"first (it writes this file with the frozen robust_eps).")
    norm_config = json.loads(normalization_config_path.read_text())
    eps = float(norm_config["robust_eps"])
    print(f"Using Run09's robust_eps={eps} (from {normalization_config_path}, computed from "
          f"TRAINING data only during train_exp05_run09.py).")

    if args.reuse_predictions:
        if not predictions_csv_path.exists():
            raise SystemExit(f"BLOCKED: --reuse-predictions given but {predictions_csv_path} does not exist.")
        with open(predictions_csv_path, newline="") as f:
            predictions = [{"sample_id": r["sample_id"], "label": float(r["label"]),
                             "probability": float(r["probability"]), "filename": r["filename"],
                             "patch_path": r["patch_path"]} for r in csv.DictReader(f)]
        checkpoint_sha256 = sha256_of_file(checkpoint_path) if checkpoint_path.exists() else "<checkpoint not found>"
        print(f"Reusing existing predictions: {predictions_csv_path} ({len(predictions)} rows)")
    else:
        if not checkpoint_path.exists():
            raise SystemExit(f"BLOCKED: checkpoint not found: {checkpoint_path}")
        checkpoint_sha256 = sha256_of_file(checkpoint_path)
        manifest_by_filename = load_manifest_by_filename(manifest_csv_path)
        all_patches, counts_by_label = scan_and_validate_test_patches(patches_dir, manifest_by_filename, test_sample_ids)
        print(f"Scanned: {counts_by_label['positive']} positive, {counts_by_label['negative']} negative "
              f"({len(all_patches)} total)")

        device = torch.device(args.device)
        model = CellCNN().to(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"Checkpoint loaded: epoch={ckpt.get('epoch')} val_pr_auc={ckpt.get('val_pr_auc')}")

        predictions = run_inference(all_patches, model, device, args.batch_size, eps, args.progress_every)
        with open(predictions_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sample_id", "label", "probability", "filename", "patch_path"])
            w.writeheader(); w.writerows(predictions)
        print(f"Saved: {predictions_csv_path}")

    y_true = np.array([p["label"] for p in predictions], dtype=np.float32)
    probs = np.array([p["probability"] for p in predictions], dtype=np.float64)
    preds_at_50 = (probs >= DEFAULT_THRESHOLD).astype(int)

    roc_auc = float(roc_auc_score(y_true, probs)) if len(set(y_true.tolist())) > 1 else float("nan")
    pr_auc = float(average_precision_score(y_true, probs))
    balanced_acc = float(balanced_accuracy_score(y_true, preds_at_50))
    acc = float(accuracy_score(y_true, preds_at_50))
    prec = float(precision_score(y_true, preds_at_50, zero_division=0))
    rec = float(recall_score(y_true, preds_at_50, zero_division=0))
    f1 = float(f1_score(y_true, preds_at_50, zero_division=0))
    cm = confusion_matrix(y_true, preds_at_50).tolist()
    threshold_table = [compute_confusion_at_threshold(y_true, probs, t) for t in THRESHOLDS]
    best_f1_row = max(threshold_table, key=lambda r: (r["f1"] if not np.isnan(r["f1"]) else -1))

    print(f"\n--- Overall test metrics ({len(predictions)} patches) ---")
    print(f"ROC-AUC={roc_auc:.6f} PR-AUC={pr_auc:.6f} BalancedAcc={balanced_acc:.4f} "
          f"Acc={acc:.4f} Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")
    print(f"Confusion matrix (threshold=0.50): {cm}")
    print(f"Best-F1 threshold: {best_f1_row['threshold']} (F1={best_f1_row['f1']:.4f})")

    diagnostic = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir), "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256, "robust_eps": eps,
        "n_patches": len(predictions),
        "n_positive": int((y_true == 1.0).sum()), "n_negative": int((y_true == 0.0).sum()),
        "roc_auc": roc_auc, "pr_auc": pr_auc, "balanced_accuracy": balanced_acc, "accuracy": acc,
        "precision": prec, "recall": rec, "f1": f1, "confusion_matrix_at_0.50": cm,
        "threshold_table": threshold_table, "best_f1": best_f1_row,
    }
    diagnostic_json_path.write_text(json.dumps(diagnostic, indent=2))
    diagnostic_txt_path.write_text(
        f"RUN09 TEST DIAGNOSTIC\nn={len(predictions)} pos={int((y_true==1.0).sum())} "
        f"neg={int((y_true==0.0).sum())}\nROC-AUC={roc_auc:.6f} PR-AUC={pr_auc:.6f} "
        f"BalancedAcc={balanced_acc:.4f} F1={f1:.4f} Precision={prec:.4f} Recall={rec:.4f}\n"
        f"Confusion matrix @0.50: {cm}\n"
    )
    print(f"\nSaved: {diagnostic_json_path}\nSaved: {diagnostic_txt_path}")
    print("\nNEXT STEP: run diagnose_matched_contrast.py against this run-dir AND Run08's "
          "run-dir with the same flags, to get the directly comparable matched-contrast numbers.")


if __name__ == "__main__":
    main()
