"""
Experiment #5, Run09: raw-contrast augmentation + robust (data-derived) normalization epsilon +
focal loss, targeting the contrast-dependence failure mode identified in Run08's matched-contrast
diagnostic (Run08: matched-contrast win rate 50.45%, |Pearson r(probability, raw_std)|=0.46).

See RUN09_DESIGN_REPORT.md for the full root-cause analysis and justification. Summary of the
THREE targeted changes vs Run08 (everything else -- split, architecture, optimizer, checkpoint
selection, balanced sampler, spatial augmentations -- is held fixed and reused unmodified):

  A) Raw-contrast augmentation, applied to the RAW patch BEFORE normalization, every epoch, via
     a new Run09-local dataset class (ContrastAugmentedTrainDataset below). This does NOT modify
     src/dataset.py::PatchDataset -- Run08's normalize-once-upfront pipeline makes contrast
     augmentation structurally impossible (by the time PatchDataset augments, every patch is
     already std=1 by construction), so Run09 moves normalization inside __getitem__ instead.
     src.augmentations.augment_patch (flip/rotate/brightness/noise) still runs afterward,
     UNMODIFIED, exactly as in Run08 -- see that module's own docstring: it expects an
     already-normalized patch, which is what it gets here too, just computed later in the pipeline.

  B) robust_eps: instead of Run08's fixed NORM_EPS=1e-6, Run09 computes
     robust_eps = max(1e-6, NOISE_FLOOR_FRACTION * median(train_raw_stds)) from TRAINING patches
     ONLY, once, frozen, and applied identically to train/val/test (same no-leakage discipline as
     every other normalization decision already made in this pipeline). Directly targets the
     epsilon-amplification mechanism identified in this session's own cross-sample-shift
     diagnostics.

  C) Focal loss (src.losses.FocalLoss via src.losses.get_criterion("focal")) -- REUSED, not
     reimplemented; it already exists in the codebase and was simply never enabled (Run08 used
     BCEWithLogitsLoss). alpha=0.5 (no extra class reweighting -- the existing balanced sampler
     already handles the 111:16476 imbalance); gamma is the sole active lever, focusing gradient
     on hard/misclassified examples.

Does NOT modify: train_exp05_run08.py, train_exp05_run07.py, diagnose_run08_test.py,
diagnose_cross_sample_shift.py, analyze_anomalous_sample_ranking.py,
visualize_anomalous_sample_patches.py, src/model.py, src/dataset.py, src/augmentations.py,
src/losses.py, src/train.py, src/experiment.py, src/config.py, manifest.csv, split_manifest.csv,
or any Run08 output. Writes ONLY under a brand-new run09_* output directory.

Usage:
    python scripts/benchmark_v2/train_exp05_run09.py \\
        --patches-dir results/exp05_training_dataset/run02_20samples/patches \\
        --split-manifest results/exp05_training_dataset/run02_20samples/split_manifest.csv \\
        --out-dir results/exp05_training_dataset/run02_20samples/run09_contrast_aug_focal_robust_eps \\
        --dry-run

    python scripts/benchmark_v2/train_exp05_run09.py \\
        --patches-dir results/exp05_training_dataset/run02_20samples/patches \\
        --split-manifest results/exp05_training_dataset/run02_20samples/split_manifest.csv \\
        --out-dir results/exp05_training_dataset/run02_20samples/run09_contrast_aug_focal_robust_eps \\
        --epochs 20 --batch-size 16 --lr 1e-3 --seed 42 --focal-gamma 2.0
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score

from src.model import CellCNN                                     # unmodified
from src.augmentations import augment_patch                       # unmodified, reused as-is
from src.losses import FocalLoss                                  # unmodified, reused (existing, unused-until-now code)
from src.train import validate, collect_predictions               # unmodified, reused
from src.experiment import Experiment                              # unmodified

LABEL_DIRS = {"positive": 1.0, "negative": 0.0}
SPLITS = ["train", "val", "test"]
RESERVED_OUT_DIR_NAMES = {"patches", "inspection"}

NORM_EPS_FLOOR = 1e-6           # absolute floor -- robust_eps never goes below Run08's own eps
NOISE_FLOOR_FRACTION_DEFAULT = 0.02
CONTRAST_SCALE_MIN_DEFAULT = 0.5
CONTRAST_SCALE_MAX_DEFAULT = 1.8
FOCAL_GAMMA_DEFAULT = 2.0
FOCAL_ALPHA_DEFAULT = 0.5        # 0.5 = no extra class reweighting; balanced sampler already handles imbalance


# --------------------------------------------------------------------------
# Provenance (identical helpers to train_exp05_run08.py, duplicated intentionally so this
# script is fully self-contained and never silently depends on that file's internals changing)
# --------------------------------------------------------------------------

def sha256_of_file(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception as e:
        return f"<could not hash: {e}>"


def extract_detector_sha256_from_summary(dataset_summary_path):
    try:
        text = Path(dataset_summary_path).read_text()
    except Exception:
        return "<dataset_summary.txt not found or unreadable>"
    m = re.search(r"adaptive_detector\.py SHA-256\s*=\s*([0-9a-f]+)", text)
    return m.group(1) if m else "<not found in dataset_summary.txt>"


# --------------------------------------------------------------------------
# Manifest / split-manifest (same logic as train_exp05_run08.py, duplicated for the same
# self-containment reason -- this file never imports from train_exp05_run08.py)
# --------------------------------------------------------------------------

def load_split_manifest(path):
    assignment = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sid, split = row["sample_id"], row["split"]
            if sid in assignment and assignment[sid] != split:
                raise ValueError(
                    f"CONTRADICTORY split_manifest.csv: sample_id '{sid}' is assigned to both "
                    f"'{assignment[sid]}' and '{split}' in different rows. Stopping."
                )
            assignment[sid] = split
    return assignment


def load_manifest_by_filename(path):
    by_filename = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            pp = row.get("patch_path", "")
            if pp:
                by_filename[Path(pp).name] = row
    return by_filename


def verify_no_leakage(split_assignment):
    by_split = {s: set() for s in SPLITS}
    for sample_id, split in split_assignment.items():
        if split not in by_split:
            raise ValueError(
                f"split_manifest.csv assigns sample '{sample_id}' to unknown split '{split}' "
                f"(expected one of {SPLITS}). Stopping."
            )
        by_split[split].add(sample_id)
    for a in SPLITS:
        for b in SPLITS:
            if a >= b:
                continue
            overlap = by_split[a] & by_split[b]
            if overlap:
                raise ValueError(
                    f"LEAKAGE DETECTED: sample(s) {sorted(overlap)} appear in BOTH '{a}' and "
                    f"'{b}' per split_manifest.csv. Stopping -- refusing to train."
                )
    return by_split


# --------------------------------------------------------------------------
# Tier 1 (fast, metadata-only, no np.load) -- identical discipline to train_exp05_run08.py
# --------------------------------------------------------------------------

def tier1_validate_filenames(patches_dir, split_name, manifest_by_filename, sample_ids_for_split):
    split_dir = Path(patches_dir) / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Required directory missing: {split_dir}")

    files_by_label = {}
    for label_name in LABEL_DIRS:
        label_dir = split_dir / label_name
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Required directory missing: {label_dir}")

        files = sorted(label_dir.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No .npy files found in {label_dir}")

        validated = []
        for f in files:
            row = manifest_by_filename.get(f.name)
            if row is None:
                raise ValueError(
                    f"LEAKAGE/INTEGRITY CHECK FAILED: {f} exists on disk but has no matching "
                    f"row in manifest.csv (by filename). Stopping."
                )
            if row["label"] != label_name:
                raise ValueError(
                    f"LABEL MISMATCH: {f} is under a '{label_name}' directory, but manifest.csv "
                    f"records its label as '{row['label']}'. Stopping."
                )
            if row["label"] == "ambiguous":
                raise ValueError(f"REFUSING TO LOAD AMBIGUOUS PATCH: {f}. Stopping.")
            if row["split"] != split_name:
                raise ValueError(
                    f"SPLIT MISMATCH: {f} is under patches/{split_name}/, but manifest.csv "
                    f"records its split as '{row['split']}'. Stopping."
                )
            if row["sample_id"] not in sample_ids_for_split:
                raise ValueError(
                    f"LEAKAGE DETECTED: {f} belongs to sample_id '{row['sample_id']}', which "
                    f"split_manifest.csv does NOT assign to split '{split_name}'. Stopping."
                )
            validated.append((f, row["sample_id"]))

        files_by_label[label_name] = validated

    return files_by_label


def _validate_patch_array(arr, f, patch_size):
    if arr.shape != (patch_size, patch_size):
        raise ValueError(f"SHAPE MISMATCH: {f} has shape {arr.shape}, expected ({patch_size}, {patch_size}).")
    if arr.dtype != np.float32:
        raise ValueError(f"DTYPE MISMATCH: {f} has dtype {arr.dtype}, expected float32. Refusing to cast.")
    if np.isnan(arr).any():
        raise ValueError(f"NaN VALUES FOUND in {f}.")
    if np.isinf(arr).any():
        raise ValueError(f"Inf VALUES FOUND in {f}.")


def load_split_with_sample_ids(files_by_label, patch_size, split_name, progress_every=5000):
    patches, labels, sample_ids = [], [], []
    total = sum(len(v) for v in files_by_label.values())
    loaded_so_far = 0

    print(f"Loading {split_name} patches (integrated structural validation)...")
    for label_name, label_value in LABEL_DIRS.items():
        for f, sample_id in files_by_label[label_name]:
            arr = np.load(f)
            _validate_patch_array(arr, f, patch_size)
            patches.append(arr)
            labels.append(label_value)
            sample_ids.append(sample_id)

            loaded_so_far += 1
            if progress_every and (loaded_so_far % progress_every == 0 or loaded_so_far == total):
                print(f"loaded {loaded_so_far} / {total}")

    print(f"{split_name.capitalize()} patch loading complete.")
    X = np.array(patches, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    return X, y, np.array(sample_ids)


# --------------------------------------------------------------------------
# Normalization -- SAME formula as Run08's apply_patch_instance_normalization, but with
# robust_eps instead of a fixed 1e-6 (intervention B). Kept as a standalone function so both
# the training dataset (per-item) and the val/test preprocessing (batched) call the identical
# computation.
# --------------------------------------------------------------------------

def instance_normalize_single(patch, eps):
    """(patch - patch.mean()) / (patch.std() + eps), computed from THIS patch's own pixels only
    -- identical formula to Run08's apply_patch_instance_normalization, just parameterized by
    eps instead of a hardcoded constant, and operating on one patch instead of a batch."""
    patch64 = patch.astype(np.float64)
    mean = patch64.mean()
    std = patch64.std()
    normed = (patch64 - mean) / (std + eps)
    return normed.astype(np.float32)


def instance_normalize_batch(X, eps):
    """Batched version, used for val/test (no augmentation -- normalize once, upfront, same as
    Run08's val/test handling)."""
    X64 = X.astype(np.float64)
    means = X64.mean(axis=(1, 2), keepdims=True)
    stds = X64.std(axis=(1, 2), keepdims=True)
    normed = (X64 - means) / (stds + eps)
    return normed.astype(np.float32)


def compute_robust_eps(X_train_raw, noise_floor_fraction):
    """Computed from TRAINING raw patches ONLY -- no val/test pixels or labels of any kind
    participate. Returns (robust_eps, median_train_raw_std) for logging/config."""
    X64 = X_train_raw.astype(np.float64)
    per_patch_std = X64.std(axis=(1, 2))
    median_std = float(np.median(per_patch_std))
    robust_eps = max(NORM_EPS_FLOOR, noise_floor_fraction * median_std)
    return robust_eps, median_std


# --------------------------------------------------------------------------
# Intervention A: raw-contrast augmentation, applied BEFORE normalization
# --------------------------------------------------------------------------

def random_contrast_jitter(raw_patch, scale_min, scale_max, rng):
    """Multiplicative rescale of pixel deviations around the RAW patch's own mean:
        patch_aug = (patch - patch.mean()) * scale + patch.mean()
    scale drawn uniformly from [scale_min, scale_max] via the given numpy Generator (explicit
    rng, not global np.random state, so this is reproducible given a seed). Purely a per-pixel
    affine rescale -- spatial layout is untouched, so morphology is preserved exactly; only the
    original contrast level changes, which is precisely the nuisance variable Run09 is trying to
    decouple from the label."""
    scale = rng.uniform(scale_min, scale_max)
    mean = raw_patch.mean()
    return ((raw_patch - mean) * scale + mean).astype(raw_patch.dtype)


class ContrastAugmentedTrainDataset(Dataset):
    """Run09-local training dataset. Holds RAW (NOT pre-normalized) patches. Per __getitem__:
        1. random_contrast_jitter() on the RAW patch (intervention A)
        2. instance_normalize_single() with the frozen robust_eps (intervention B)
        3. augment_patch() from src.augmentations, UNMODIFIED -- same flip/rotate/brightness/
           noise stack Run08 used, applied to the now-normalized patch exactly as that module's
           docstring specifies.
    This intentionally does NOT subclass or modify src/dataset.py::PatchDataset -- that class's
    design (normalize the whole array once upfront) is structurally incompatible with varying
    raw contrast per-epoch, so this is a new, separate, Run09-only class instead of a patch to
    shared code."""

    def __init__(self, X_raw, y, robust_eps, contrast_scale_min, contrast_scale_max, seed):
        self.X_raw = X_raw
        self.y = y
        self.robust_eps = robust_eps
        self.contrast_scale_min = contrast_scale_min
        self.contrast_scale_max = contrast_scale_max
        # one Generator per dataset instance; PyTorch DataLoader workers (if any) each get a
        # forked copy, which is fine for a diagnostic-grade augmentation (not used for anything
        # requiring cross-worker independence guarantees).
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X_raw)

    def __getitem__(self, idx):
        raw_patch = self.X_raw[idx]
        if raw_patch.ndim == 3:
            raw_patch = raw_patch[0]

        jittered = random_contrast_jitter(raw_patch, self.contrast_scale_min, self.contrast_scale_max, self._rng)
        normalized = instance_normalize_single(jittered, self.robust_eps)
        augmented = augment_patch(normalized)  # unmodified src.augmentations.augment_patch

        patch_t = torch.tensor(augmented, dtype=torch.float32).unsqueeze(0)
        label_t = torch.tensor(self.y[idx], dtype=torch.float32)
        return patch_t, label_t


class NormalizedEvalDataset(Dataset):
    """Val/test dataset: pre-normalized ONCE upfront (via instance_normalize_batch), no
    augmentation, no contrast jitter -- identical evaluation discipline to Run08's val/test
    handling, just using robust_eps instead of the fixed 1e-6."""

    def __init__(self, X_normalized, y):
        self.X = X_normalized
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        patch = self.X[idx]
        if patch.ndim == 3:
            patch = patch[0]
        patch_t = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
        label_t = torch.tensor(self.y[idx], dtype=torch.float32)
        return patch_t, label_t


# --------------------------------------------------------------------------
# Custom train_one_epoch (mirrors src.train.train_one_epoch exactly, duplicated only because
# that function is not parameterized to accept an already-built criterion+loader in a way this
# script needs differently -- behavior is otherwise identical: same optimizer step order, same
# accuracy computation via logit>=0)
# --------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predictions = (outputs >= 0).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return running_loss / len(loader), correct / total


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches-dir", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--manifest-csv", default=None, help="Default: <patches-dir>/../manifest.csv")
    parser.add_argument("--dataset-summary", default=None, help="Default: <patches-dir>/../dataset_summary.txt")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-log-every", type=int, default=5000)
    parser.add_argument("--noise-floor-fraction", type=float, default=NOISE_FLOOR_FRACTION_DEFAULT)
    parser.add_argument("--contrast-scale-min", type=float, default=CONTRAST_SCALE_MIN_DEFAULT)
    parser.add_argument("--contrast-scale-max", type=float, default=CONTRAST_SCALE_MAX_DEFAULT)
    parser.add_argument("--focal-gamma", type=float, default=FOCAL_GAMMA_DEFAULT)
    parser.add_argument("--focal-alpha", type=float, default=FOCAL_ALPHA_DEFAULT)
    parser.add_argument("--allow-nonempty-out-dir", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                         help="Validate + compute robust_eps + one forward pass; no training, "
                              "no backward(), no optimizer.step(), no checkpoint")
    args = parser.parse_args()

    patches_dir = Path(args.patches_dir)
    stage1_root = patches_dir.parent
    manifest_csv_path = Path(args.manifest_csv) if args.manifest_csv else stage1_root / "manifest.csv"
    dataset_summary_path = Path(args.dataset_summary) if args.dataset_summary else stage1_root / "dataset_summary.txt"
    split_manifest_path = Path(args.split_manifest)
    out_dir = Path(args.out_dir)

    if out_dir.name in RESERVED_OUT_DIR_NAMES:
        raise SystemExit(f"Refusing to use reserved directory name '{out_dir.name}' as --out-dir.")
    if out_dir.resolve() == stage1_root.resolve():
        raise SystemExit("Refusing to write training artifacts into the Stage 1 root directory itself.")
    if "run08" in out_dir.name.lower():
        raise SystemExit(f"Refusing to write Run09 artifacts into a directory named '{out_dir.name}' "
                          f"-- this looks like it could collide with Run08. Use a run09_* name.")
    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_nonempty_out_dir:
        raise SystemExit(
            f"Refusing to write into non-empty existing directory {out_dir} "
            f"(pass --allow-nonempty-out-dir to override)."
        )
    for protected in ["run09_config.json", "normalization_config.json", "training_metrics.csv"]:
        if (out_dir / protected).exists():
            raise SystemExit(f"Refusing to overwrite existing {out_dir / protected}.")

    run_timestamp_utc = datetime.now(timezone.utc).isoformat()
    print(f"Experiment #5 Run09 {'DRY RUN' if args.dry_run else 'TRAINING'}: "
          f"contrast augmentation + robust eps + focal loss")
    print(f"patches_dir={patches_dir}\nmanifest_csv={manifest_csv_path}\nsplit_manifest={split_manifest_path}\n")

    for required in [patches_dir, split_manifest_path, manifest_csv_path]:
        if not required.exists():
            raise SystemExit(f"BLOCKED: required path does not exist: {required}")

    # ---- frozen split: loaded, never recomputed ----
    split_assignment = load_split_manifest(split_manifest_path)
    by_split_sample_ids = verify_no_leakage(split_assignment)
    print("Leakage precheck: train/val/test sample IDs are disjoint (from split_manifest.csv). "
          "Loaded, not recomputed -- identical split to Run08.")
    for s in SPLITS:
        print(f"  {s}: {len(by_split_sample_ids[s])} samples -> {sorted(by_split_sample_ids[s])}")

    manifest_by_filename = load_manifest_by_filename(manifest_csv_path)
    print(f"\nLoaded manifest.csv: {len(manifest_by_filename)} rows with a saved patch_path.")

    print("\nTier 1 metadata validation (no np.load() calls)...")
    files_by_split_label = {}
    for split_name in ["train", "val"]:
        files_by_split_label[split_name] = tier1_validate_filenames(
            patches_dir, split_name, manifest_by_filename, by_split_sample_ids[split_name],
        )
        n_pos = len(files_by_split_label[split_name]["positive"])
        n_neg = len(files_by_split_label[split_name]["negative"])
        print(f"  {split_name}: {n_pos} positive, {n_neg} negative ({n_pos + n_neg} total)")
    print("Tier 1 complete.")

    print("\nTier 2: integrated single-pass load (train + val only; test is a separate "
          "evaluation script's job, same as Run08)...")
    X_train_raw, y_train, sids_train = load_split_with_sample_ids(
        files_by_split_label["train"], args.patch_size, "train", args.progress_log_every)
    X_val_raw, y_val, sids_val = load_split_with_sample_ids(
        files_by_split_label["val"], args.patch_size, "val", args.progress_log_every)

    # ---- intervention B: robust_eps from TRAINING data only ----
    robust_eps, median_train_raw_std = compute_robust_eps(X_train_raw, args.noise_floor_fraction)
    print(f"\nrobust_eps = max({NORM_EPS_FLOOR}, {args.noise_floor_fraction} * "
          f"median_train_raw_std={median_train_raw_std:.4f}) = {robust_eps:.6f}  "
          f"(Run08's fixed NORM_EPS was 1e-6)")

    # val is normalized once, upfront, with robust_eps -- no augmentation, matches Run08's
    # val/test handling discipline exactly (just eps differs).
    X_val = instance_normalize_batch(X_val_raw, robust_eps)

    out_dir.mkdir(parents=True, exist_ok=True)
    norm_config_path = out_dir / "normalization_config.json"
    norm_config_path.write_text(json.dumps({
        "method": "patch-level instance normalization with a robust, training-data-derived "
                  "epsilon (Run09 intervention B), plus raw-contrast augmentation applied "
                  "before normalization during training only (Run09 intervention A)",
        "formula": "(patch.astype(np.float64) - patch.mean()) / (patch.std() + robust_eps)",
        "robust_eps": robust_eps,
        "norm_eps_floor": NORM_EPS_FLOOR,
        "noise_floor_fraction": args.noise_floor_fraction,
        "median_train_raw_std": median_train_raw_std,
        "robust_eps_computed_from": "TRAINING raw patches ONLY -- no validation or test pixel "
                                     "values or labels of any kind participate in this "
                                     "computation.",
        "contrast_augmentation": {
            "applied_to": "TRAINING patches only, before normalization, every epoch (fresh "
                           "random scale per item per epoch)",
            "formula": "(patch - patch.mean()) * scale + patch.mean(), scale ~ "
                       "Uniform(contrast_scale_min, contrast_scale_max)",
            "contrast_scale_min": args.contrast_scale_min,
            "contrast_scale_max": args.contrast_scale_max,
            "preserves_morphology": "exactly -- purely a per-pixel amplitude rescale around the "
                                     "patch's own mean; no spatial resampling of any kind.",
        },
        "identical_across_train_val_test": "the underlying instance-normalization FORMULA and "
            "robust_eps are identical for train/val/test; contrast augmentation is applied ONLY "
            "to training patches (never validation or test).",
    }, indent=2))
    print(f"Saved: {norm_config_path}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds = ContrastAugmentedTrainDataset(
        X_train_raw, y_train, robust_eps, args.contrast_scale_min, args.contrast_scale_max, args.seed,
    )
    val_ds = NormalizedEvalDataset(X_val, y_val)

    n_positive = int(np.sum(y_train == 1.0))
    n_negative = int(np.sum(y_train == 0.0))
    if n_positive == 0 or n_negative == 0:
        raise SystemExit(f"BLOCKED: cannot create balanced sampler: n_positive={n_positive}, n_negative={n_negative}")
    sample_weights = np.where(y_train == 1.0, 1.0 / n_positive, 1.0 / n_negative).astype(np.float64)
    train_sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(y_train), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"\nBalanced training sampler (kept UNCHANGED from Run06/07/08 -- see design report "
          f"for why this is not doubled up with focal loss's alpha term): {n_positive} positive "
          f"/ {n_negative} negative originals (positive rate={n_positive/len(y_train):.4%}); "
          f"validation loader remains at its original distribution.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    check_sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=args.batch_size, replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    check_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=check_sampler, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = CellCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ---- intervention C: focal loss, reused from src.losses, alpha=0.5 (no extra class weight) ----
    criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    print(f"Model: CellCNN (unmodified). Loss: FocalLoss(gamma={args.focal_gamma}, "
          f"alpha={args.focal_alpha}) -- reused from src.losses, enabled for the first time in "
          f"this experiment lineage (Run06/07/08 all used BCEWithLogitsLoss).")

    # ---- forward-pass compatibility check (dry-run and full-run alike) ----
    model.eval()
    params_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        images, labels = next(iter(check_loader))
        images, labels = images.to(device), labels.to(device)
        logits = model(images).squeeze(1)
        loss_value = criterion(logits, labels).item()
    params_after = {n: p.detach().clone() for n, p in model.named_parameters()}
    all_unchanged = all(torch.equal(params_before[n], params_after[n]) for n in params_before)
    print(f"\nForward-pass compatibility check: images.shape={tuple(images.shape)} "
          f"loss={loss_value} finite={np.isfinite(loss_value)} params_unchanged={all_unchanged}")
    if not np.isfinite(loss_value):
        raise SystemExit("BLOCKED: forward-pass loss is not finite. Stopping.")
    if not all_unchanged:
        raise SystemExit("BLOCKED: model parameters changed during a supposedly no-grad check. Stopping.")

    detector_sha256 = extract_detector_sha256_from_summary(dataset_summary_path)

    config_record = {
        "experiment_name": out_dir.name,
        "run_timestamp_utc": run_timestamp_utc,
        "command_line": " ".join(sys.argv),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "loss": f"FocalLoss(gamma={args.focal_gamma}, alpha={args.focal_alpha})  [Run08 used BCEWithLogitsLoss(pos_weight=1.0)]",
        "balanced_train_sampler": True,
        "normalization_method": "robust patch-level instance normalization -- see normalization_config.json",
        "contrast_augmentation": {"scale_min": args.contrast_scale_min, "scale_max": args.contrast_scale_max},
        "checkpoint_selection_metric": "val_pr_auc",
        "early_stopping_metric": "val_pr_auc",
        "lr_scheduler": "ReduceLROnPlateau (monitors val_loss, unchanged from Run06/07/08)",
        "model_name": "CellCNN",
        "model_n_parameters": n_params,
        "model_n_trainable_parameters": n_trainable,
        "stage1_source_dir": str(stage1_root),
        "stage1_manifest_path": str(manifest_csv_path),
        "stage1_split_manifest_path": str(split_manifest_path),
        "stage1_detector_sha256": detector_sha256,
        "train_samples": sorted(by_split_sample_ids["train"]),
        "val_samples": sorted(by_split_sample_ids["val"]),
        "test_samples": sorted(by_split_sample_ids["test"]),
        "train_patches": {"positive": n_positive, "negative": n_negative},
        "val_patches": {"positive": int(np.sum(y_val == 1.0)), "negative": int(np.sum(y_val == 0.0))},
        "normalization_config_path": str(norm_config_path),
        "changes_vs_run08": [
            "A) raw-contrast augmentation before normalization (training only)",
            "B) robust_eps computed from training data (vs Run08's fixed 1e-6)",
            "C) FocalLoss (reused from src.losses) instead of BCEWithLogitsLoss",
        ],
        "unchanged_vs_run08": [
            "split (loaded from split_manifest.csv, never recomputed)",
            "architecture (CellCNN)", "optimizer (Adam)", "LR scheduler (ReduceLROnPlateau on val_loss)",
            "checkpoint selection metric (val_pr_auc)", "balanced training sampler",
            "spatial augmentations (flip, 90-degree rotation, from src.augmentations, unmodified)",
        ],
        "dry_run": args.dry_run,
    }

    print(f"\n--- SANITY CHECK (before any training) ---")
    print(f"Model architecture: CellCNN (3 conv blocks + BN + dropout=0.3)")
    print(f"Trainable parameters: {n_trainable:,} / {n_params:,} total")
    print(f"Train: {len(y_train)} patches ({n_positive} positive / {n_negative} negative)")
    print(f"Val:   {len(y_val)} patches ({int(np.sum(y_val==1.0))} positive / {int(np.sum(y_val==0.0))} negative)")
    print(f"Test:  {len(by_split_sample_ids['test'])} samples (evaluated separately, not loaded here)")
    print(f"Normalization: robust patch-instance norm, robust_eps={robust_eps:.6f}")
    print(f"Contrast augmentation: scale ~ U({args.contrast_scale_min}, {args.contrast_scale_max}), train only")
    print(f"Spatial/other augmentation: src.augmentations.augment_patch (unmodified)")
    print(f"Loss: FocalLoss(gamma={args.focal_gamma}, alpha={args.focal_alpha})")
    print(f"Optimizer: Adam(lr={args.lr})")
    print(f"Batch size: {args.batch_size}")
    print(f"Seed: {args.seed}")
    print(f"Output directory: {out_dir}")
    print("--- END SANITY CHECK ---\n")

    if args.dry_run:
        (out_dir / "run09_config.json").write_text(json.dumps(config_record, indent=2))
        print(f"\nDRY RUN COMPLETE. No training performed. No checkpoint created.")
        print(f"Config written to {out_dir / 'run09_config.json'}")
        return

    # ============================================================
    # Full training
    # ============================================================
    exp = Experiment(base_dir=out_dir.parent, name=out_dir.name)
    exp.save_config(extra=config_record)
    (out_dir / "run09_config.json").write_text(json.dumps(config_record, indent=2))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    metrics_rows = []
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_pr_auc = -float("inf")
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        y_val_true, y_val_prob = collect_predictions(model, val_loader, device)
        val_roc_auc = float(roc_auc_score(y_val_true, y_val_prob)) if len(set(y_val_true.tolist())) > 1 else float("nan")
        val_pr_auc = float(average_precision_score(y_val_true, y_val_prob))
        scheduler.step(val_loss)

        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss={train_loss:.4f} Acc={train_acc:.3f} "
              f"| Val Loss={val_loss:.4f} Acc={val_acc:.3f} ROC-AUC={val_roc_auc:.4f} "
              f"PR-AUC={val_pr_auc:.4f} | LR={lr_now:.2e}")
        exp.log_epoch(epoch + 1, train_loss, train_acc, val_loss, val_acc, lr_now)
        metrics_rows.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_acc,
            "val_loss": val_loss, "val_accuracy": val_acc, "val_roc_auc": val_roc_auc,
            "val_pr_auc": val_pr_auc, "learning_rate": lr_now,
        })

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch,
                "val_loss": val_loss, "val_accuracy": val_acc,
                "val_roc_auc": val_roc_auc, "val_pr_auc": val_pr_auc,
            }, exp.dir / "model.pth")
            print(f"  New best model saved (val_pr_auc={val_pr_auc:.4f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stopping_patience:
            print("Validation PR-AUC stopped improving. Stopping...")
            break

    exp.save_metrics_csv()
    exp.save_plots(train_losses, val_losses, train_accs, val_accs)

    with open(out_dir / "training_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss",
                                           "val_accuracy", "val_roc_auc", "val_pr_auc", "learning_rate"])
        w.writeheader()
        w.writerows(metrics_rows)
    print(f"\nSaved: {out_dir / 'training_metrics.csv'}")

    state = torch.load(exp.dir / "model.pth", map_location=device, weights_only=False)
    checkpoint_sha256 = sha256_of_file(exp.dir / "model.pth")
    print(f"\nBest checkpoint: epoch={state['epoch']} val_loss={state['val_loss']:.4f} "
          f"val_accuracy={state['val_accuracy']:.4f} val_roc_auc={state['val_roc_auc']:.4f} "
          f"val_pr_auc={state['val_pr_auc']:.4f}")
    print(f"checkpoint SHA-256: {checkpoint_sha256}")
    (out_dir / "checkpoint_sha256.txt").write_text(checkpoint_sha256 + "\n")

    exp.promote_to_default()
    print(f"\nTraining complete: {exp.dir}")
    print(f"\nNEXT STEP: evaluate on the held-out test set (reuse the diagnose_run08_test.py "
          f"PATTERN -- normalize test patches with robust_eps={robust_eps:.6f} from "
          f"{norm_config_path}, run inference, save test_predictions_resumable.csv, then run "
          f"diagnose_matched_contrast.py against this run's output directory).")


if __name__ == "__main__":
    main()
