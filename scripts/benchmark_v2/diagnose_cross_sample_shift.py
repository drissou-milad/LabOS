"""
Experiment #5: cross-sample shift diagnostic (Run08 post-hoc analysis).

DIAGNOSTIC ONLY. Trains nothing, loads no optimizer, touches no checkpoint's weights (it does
not even need to load the model -- Run08's predictions already exist in
test_predictions_resumable.csv). Does not modify, and does not need to modify:

    scripts/benchmark_v2/train_exp05_run08.py
    scripts/benchmark_v2/diagnose_run08_test.py
    scripts/benchmark_v2/train_exp05_run07.py
    scripts/benchmark_v2/diagnose_run07_test.py
    results/exp05_training_dataset/run02_20samples/manifest.csv
    results/exp05_training_dataset/run02_20samples/split_manifest.csv
    results/exp05_training_dataset/run02_20samples/run08_patch_instance_norm/model.pth
    results/exp05_training_dataset/run02_20samples/run08_patch_instance_norm/test_predictions_resumable.csv

WHY THIS SCRIPT EXISTS: Run08's pooled test ROC-AUC (0.612928) looks like a mild improvement
over Run07, but the per-sample breakdown is not "three mediocre samples" -- it is one sample
(44b6_0b24845f) that appears INVERTED (ROC-AUC=0.101986, well below the 0.5 chance line, with
positive mean prediction 0.199034 < negative mean prediction 0.365178), and two other samples
hovering near chance (ROC-AUC 0.550934 / 0.535220). Averaging these into one pooled number
hides a qualitatively different failure mode on one sample. Before committing to a Run09 design
change, this script investigates WHY 44b6_0b24845f behaves this way, using only data that
already exists (no new model runs, no new normalization decisions).

WHAT THIS SCRIPT DOES:
    1. Loads the frozen split from split_manifest.csv (loaded, never recomputed -- this script
       has no split-assignment logic of any kind).
    2. Loads manifest.csv and cross-validates every prediction row against it (same
       filename/label/split/sample_id integrity discipline as diagnose_run08_test.py), so a
       stale or hand-edited predictions CSV cannot silently corrupt the diagnosis.
    3. Recomputes pooled and per-sample ROC-AUC/PR-AUC directly from
       test_predictions_resumable.csv and compares them against the values you reported, so any
       discrepancy is surfaced immediately rather than assumed away.
    4. Loads sample_distribution_audit.json for cross-sample RAW intensity context (train/val/
       test sample-level mean/std, as originally audited -- this file is read-only input here,
       never recomputed).
    5. HYPOTHESIS TEST -- epsilon amplification on near-flat patches: for every test patch
       referenced in test_predictions_resumable.csv, loads the RAW (pre-normalization) .npy
       array via its recorded patch_path and computes that patch's own raw mean/std (the exact
       same statistic train_exp05_run08.py's apply_patch_instance_normalization would have
       divided by). Patches whose raw std is very small relative to the rest of the test set are
       flagged as "epsilon-dominated" (their post-normalization values are disproportionately
       driven by (std + 1e-6) rather than genuine pixel contrast). Reports, per sample and per
       label: the raw-std distribution, the fraction of epsilon-dominated patches, and whether
       predicted probability correlates with raw std / raw mean -- separately for positive and
       negative patches, since a correlation reversal between labels is exactly what would
       explain an inverted ROC-AUC.
    6. Reports whether 44b6_0b24845f's raw intensity/contrast profile is an outlier relative to
       the OTHER TWO test samples and relative to the train distribution (from
       sample_distribution_audit.json), to distinguish "this sample has an unusual contrast
       profile" from "this sample just has very few positives (n=4) and is noisy".

WHAT THIS SCRIPT DOES NOT DO: it does not propose or apply a fix, does not retrain, does not
touch the Run08 checkpoint, and does not change the frozen split or any manifest. It is
read-only with respect to every file in the bullet list above. All outputs are written under a
NEW diagnostics subdirectory so nothing already produced by Run08 can be overwritten.

Usage:
    python scripts/benchmark_v2/diagnose_cross_sample_shift.py

    python scripts/benchmark_v2/diagnose_cross_sample_shift.py \\
        --stage1-root results/exp05_training_dataset/run02_20samples \\
        --run-dir results/exp05_training_dataset/run02_20samples/run08_patch_instance_norm \\
        --overwrite
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

TEST_SAMPLES = {
    "44b6_0b24845f",
    "44b6_1d530831",
    "44b6_33b596bf",
}
SPLITS = ["train", "val", "test"]

STAGE1_ROOT_DEFAULT = "results/exp05_training_dataset/run02_20samples"
RUN_DIR_DEFAULT = f"{STAGE1_ROOT_DEFAULT}/run08_patch_instance_norm"

# Reported Run08 numbers, from the diagnostic conversation -- used ONLY as a sanity-check
# comparison against what this script independently recomputes from the predictions CSV; never
# used as an input to any computation below.
REPORTED_OVERALL = {"roc_auc": 0.612928, "pr_auc": 0.009621}
REPORTED_PER_SAMPLE = {
    "44b6_0b24845f": {"roc_auc": 0.101986, "pr_auc": 0.001304,
                       "positive_mean": 0.199034, "negative_mean": 0.365178},
    "44b6_1d530831": {"roc_auc": 0.550934, "pr_auc": 0.011554},
    "44b6_33b596bf": {"roc_auc": 0.535220, "pr_auc": 0.002756},
}
METRIC_MISMATCH_TOLERANCE = 1e-3

PERCENTILES = [0, 5, 10, 25, 50, 75, 90, 95, 100]

# Independently-verified expected held-out test counts (same values used to gate
# diagnose_run08_test.py). --dry-run checks the on-disk/manifest state against these; it does
# not derive them from anything computed at runtime.
EXPECTED_TOTAL = 16587
EXPECTED_POSITIVE = 111
EXPECTED_NEGATIVE = 16476
EXPECTED_PER_SAMPLE_COUNTS = {
    "44b6_0b24845f": {"total": 2068, "positive": 4, "negative": 2064},
    "44b6_1d530831": {"total": 9763, "positive": 95, "negative": 9668},
    "44b6_33b596bf": {"total": 4756, "positive": 12, "negative": 4744},
}

# Candidate join-key columns, in preference order, used to match predictions rows to manifest
# rows during --dry-run. Detected dynamically from each file's actual header -- never assumed.
JOIN_KEY_CANDIDATES = ["patch_path", "filename", "patch_filename", "path"]

# --- Resumable raw-patch-stats checkpoint (load_raw_patch_stats) ---
RAW_STATS_CHECKPOINT_FILENAME = "raw_patch_stats_checkpoint.npz"
RAW_STATS_CHECKPOINT_META_FILENAME = "raw_patch_stats_checkpoint.meta.json"
RAW_STATS_CHECKPOINT_EVERY = 250  # save progress every N newly-computed patches


# --------------------------------------------------------------------------
# Provenance / loading (read-only; same integrity discipline as diagnose_run08_test.py)
# --------------------------------------------------------------------------

def load_split_manifest(path):
    """Loads the FROZEN split. Never recomputes, never reassigns -- a pure lookup table."""
    assignment = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sid, split = row["sample_id"], row["split"]
            if sid in assignment and assignment[sid] != split:
                raise ValueError(
                    f"CONTRADICTORY split_manifest.csv: sample_id '{sid}' assigned to both "
                    f"'{assignment[sid]}' and '{split}'. Stopping."
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


def load_predictions(predictions_csv_path):
    rows = []
    with open(predictions_csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "sample_id": r["sample_id"],
                "label": float(r["label"]),
                "probability": float(r["probability"]),
                "filename": r["filename"],
                "patch_path": r["patch_path"],
            })
    return rows


def cross_validate_predictions_against_manifest(predictions, manifest_by_filename, split_assignment):
    """Refuses to proceed if the predictions CSV disagrees with manifest.csv / split_manifest.csv
    on label, split, or sample_id for any row -- catches a stale or hand-edited predictions file
    before it can quietly corrupt this diagnosis."""
    problems = []
    for p in predictions:
        row = manifest_by_filename.get(p["filename"])
        if row is None:
            problems.append(f"{p['filename']}: not found in manifest.csv")
            continue
        manifest_label_value = 1.0 if row["label"] == "positive" else (0.0 if row["label"] == "negative" else None)
        if manifest_label_value is None or manifest_label_value != p["label"]:
            problems.append(f"{p['filename']}: predictions label={p['label']} vs manifest label={row['label']}")
        if row["split"] != "test":
            problems.append(f"{p['filename']}: manifest split={row['split']}, expected 'test'")
        if row["sample_id"] != p["sample_id"]:
            problems.append(f"{p['filename']}: predictions sample_id={p['sample_id']} vs manifest sample_id={row['sample_id']}")
        if split_assignment.get(row["sample_id"]) != "test":
            problems.append(f"{p['filename']}: sample_id {row['sample_id']} is not assigned split='test' in split_manifest.csv")
    if problems:
        preview = "\n  ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise ValueError(
            f"BLOCKED: {len(problems)} prediction row(s) disagree with manifest.csv / "
            f"split_manifest.csv. Refusing to diagnose against an inconsistent predictions "
            f"file:\n  {preview}{more}"
        )
    print(f"Cross-validated {len(predictions)} prediction rows against manifest.csv and "
          f"split_manifest.csv: all consistent.")


# --------------------------------------------------------------------------
# --dry-run helpers: validation only, no np.load() on patch arrays, no full analysis
# --------------------------------------------------------------------------

def get_csv_fieldnames(path):
    """Reads only the header row -- returns the file's ACTUAL column names. Never assumes a
    schema; every dry-run consumer of a CSV calls this first and prints the result."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    return header


def detect_join_key(manifest_fields, predictions_fields):
    """Picks the first candidate column (in JOIN_KEY_CANDIDATES preference order) that exists in
    BOTH the manifest and predictions headers. Does not assume 'filename' or 'patch_path' is
    present -- inspects the actual headers first and reports exactly which key (if any) is
    usable, and why. NOTE: this reports which COLUMN is shared for display purposes only; it does
    NOT determine how values in that column are compared -- see determine_canonical_matching()."""
    manifest_set = set(manifest_fields)
    predictions_set = set(predictions_fields)
    for candidate in JOIN_KEY_CANDIDATES:
        in_manifest = candidate in manifest_set
        in_predictions = candidate in predictions_set
        if in_manifest and in_predictions:
            return candidate, f"'{candidate}' present in both manifest.csv and predictions CSV headers"
    return None, (
        f"NONE of the candidate join keys {JOIN_KEY_CANDIDATES} are present in both headers. "
        f"manifest.csv columns={manifest_fields}  predictions columns={predictions_fields}"
    )


def path_basename(value):
    """Extracts the final path component, robust to BOTH '/' and '\\\\' separators regardless of
    the host OS. pathlib.Path(...).name is OS-dependent: on a POSIX host, PurePosixPath does not
    treat backslashes as separators at all, so a Windows-style manifest.csv path would return
    unchanged (not just its filename) and silently fail to match. Since manifest.csv and the
    predictions CSV in this pipeline are typically produced on Windows (per the project's own
    PowerShell workflow) but this script may be inspected/tested on any host, basename extraction
    here is done manually rather than trusting pathlib's platform-dependent behavior."""
    if not value:
        return ""
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def determine_canonical_matching(manifest_fields, predictions_fields):
    """Determines the join REPRESENTATION to use -- not just which column name is shared.

    The existing, already-proven full-run matching (load_manifest_by_filename() +
    cross_validate_predictions_against_manifest(), both left unmodified) does NOT compare raw
    patch_path strings. It keys manifest rows by Path(patch_path).name (the filename basename)
    and matches predictions rows via their own 'filename' column. Raw patch_path strings can
    legitimately differ between manifest.csv and a predictions CSV written at a different time /
    from a different working directory / with different path separators -- comparing them
    verbatim is what caused the previous dry-run bug (16587 predictions, 0 matches), even though
    the full run's filename-basename matching succeeds on the exact same two files.

    This function reproduces that same semantics for the dry-run's matched-set construction (via
    path_basename(), which is separator-agnostic regardless of host OS -- see path_basename's
    docstring), so the two code paths agree by construction rather than by coincidence. It falls
    back to a raw shared-column comparison only if the filename/patch_path combination isn't
    available, and always reports explicitly (via the returned 'description') which
    representation is in use -- dry-run output must never claim raw-string matching when
    canonicalization is actually being applied.
    """
    if "patch_path" in manifest_fields and "filename" in predictions_fields:
        return {
            "manifest_key_fn": lambda row: path_basename(row.get("patch_path", "")),
            "prediction_key_fn": lambda row: row.get("filename", ""),
            "join_key_label": "patch_path",
            "description": (
                "filename (basename of manifest.csv's patch_path, separator-agnostic), matched "
                "against the predictions CSV's own 'filename' column -- identical semantics to "
                "the existing, unmodified load_manifest_by_filename() + "
                "cross_validate_predictions_against_manifest() used by the full run. Raw "
                "patch_path strings are NOT compared directly."
            ),
        }
    if "patch_path" in manifest_fields and "patch_path" in predictions_fields:
        # No 'filename' column to fall back on -- canonicalize both sides to a basename so this
        # still agrees with the full run's semantics rather than comparing raw path strings
        # (which may differ in absolute/relative form or path separators).
        return {
            "manifest_key_fn": lambda row: path_basename(row.get("patch_path", "")),
            "prediction_key_fn": lambda row: path_basename(row.get("patch_path", "")),
            "join_key_label": "patch_path",
            "description": (
                "filename (basename of patch_path, separator-agnostic) on BOTH sides -- "
                "predictions CSV has no separate 'filename' column, so this canonicalizes "
                "patch_path the same way load_manifest_by_filename() does rather than comparing "
                "raw patch_path strings, which can differ in representation between files."
            ),
        }
    join_key, reason = detect_join_key(manifest_fields, predictions_fields)
    if join_key is None:
        return None
    return {
        "manifest_key_fn": lambda row, jk=join_key: row.get(jk, ""),
        "prediction_key_fn": lambda row, jk=join_key: row.get(jk, ""),
        "join_key_label": join_key,
        "description": (
            f"raw '{join_key}' column value, compared verbatim (no canonicalization applied) -- "
            f"neither a shared 'filename' column nor a shared 'patch_path' column was available "
            f"to reproduce the full run's basename-matching semantics. {reason}"
        ),
    }


def dry_run_load_split_manifest(path):
    """Same frozen-split loading/validation as the full run (load_split_manifest +
    verify_no_leakage's disjointness check), but returns per-split sample_id sets directly for
    printing, without doing anything downstream with them."""
    assignment = load_split_manifest(path)
    by_split = {s: set() for s in SPLITS}
    for sample_id, split in assignment.items():
        if split not in by_split:
            raise ValueError(
                f"split_manifest.csv assigns sample '{sample_id}' to unknown split '{split}' "
                f"(expected one of {SPLITS})."
            )
        by_split[split].add(sample_id)
    for a in SPLITS:
        for b in SPLITS:
            if a >= b:
                continue
            overlap = by_split[a] & by_split[b]
            if overlap:
                raise ValueError(f"LEAKAGE DETECTED: sample(s) {sorted(overlap)} appear in both '{a}' and '{b}'.")
    return assignment, by_split


def dry_run_manifest_counts_by_split(manifest_csv_path):
    """INFORMATIONAL ONLY. Tallies positive/negative/total counts per split directly from EVERY
    row currently in manifest.csv (label in {'positive','negative'} only). This reflects the
    manifest as it stands today, which is NOT necessarily the same set of test patches Run08 was
    actually trained/evaluated against -- manifest.csv can grow (e.g. from a later dataset-build
    pass) without the frozen Run08 prediction set changing. Do not use this function's 'test'
    counts for expected-count validation; use build_run08_matched_manifest_rows() + the
    prediction CSV instead, which is the actual authoritative definition of what Run08 saw."""
    counts = {s: {"positive": 0, "negative": 0, "total": 0} for s in SPLITS}
    with open(manifest_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            split = row.get("split")
            label = row.get("label")
            if split not in SPLITS or label not in ("positive", "negative"):
                continue
            counts[split][label] += 1
            counts[split]["total"] += 1
    return counts


# --------------------------------------------------------------------------
# Run08 prediction-linked validation: the predictions CSV, not "all manifest rows where
# split=='test'", is the authoritative definition of the frozen Run08 evaluated patch set.
# Matching uses the SAME canonical representation as the full run (see
# determine_canonical_matching / cross_validate_predictions_against_manifest), never raw
# patch_path string comparison. Prediction rows themselves are loaded via the existing,
# unmodified load_predictions() -- the same loader the full run uses -- so canonical keys AND
# patch_path values both come from that single, already-correct source.
# --------------------------------------------------------------------------

def build_run08_matched_manifest_rows(manifest_csv_path, manifest_key_fn, prediction_key_set):
    """Filters manifest.csv down to ONLY the rows whose canonical key (manifest_key_fn(row))
    appears in the Run08 prediction set's canonical keys. This -- not 'every manifest row with
    split==test', and not raw patch_path string equality -- is what expected-count validation
    must run against."""
    matched = []
    with open(manifest_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            key = manifest_key_fn(row)
            if key and key in prediction_key_set:
                matched.append(row)
    return matched


def compute_run08_matched_counts(matched_rows):
    """total/positive/negative + per-sample breakdown computed ONLY from the Run08
    prediction-linked matched manifest rows -- never from the full manifest."""
    counts = {"positive": 0, "negative": 0, "total": 0}
    per_sample = {}
    for row in matched_rows:
        label = row.get("label")
        sid = row.get("sample_id")
        d = per_sample.setdefault(sid, {"positive": 0, "negative": 0, "total": 0})
        if label in ("positive", "negative"):
            counts[label] += 1
            counts["total"] += 1
            d[label] += 1
            d["total"] += 1
    return counts, per_sample


def run08_matched_set_integrity_checks(prediction_values, matched_rows, matched_rows_key_fn, matching_label):
    """All of the checks required before trusting the matched set:
      - every prediction identifier exists in manifest.csv
      - no duplicate prediction identifiers silently collapse the expected count
      - no duplicate canonical-key values among matched manifest rows
      - matched manifest row count == prediction row count
      - every matched row has split == 'test'
      - every matched row's sample_id is one of the frozen TEST_SAMPLES
      - the matched sample_id set is EXACTLY the frozen test set (no more, no fewer)
    Returns a list of human-readable problem strings; empty list means all checks passed.
    matching_label is only used to make issue messages self-describing (e.g. 'filename')."""
    issues = []

    n_pred = len(prediction_values)
    n_pred_unique = len(set(prediction_values))
    if n_pred_unique != n_pred:
        issues.append(
            f"{n_pred - n_pred_unique} duplicate prediction identifier(s) in the predictions "
            f"CSV (matching representation: {matching_label}) -- duplicates would silently "
            f"collapse the expected count."
        )

    matched_keys = [matched_rows_key_fn(row) for row in matched_rows]
    n_matched = len(matched_keys)
    n_matched_unique = len(set(matched_keys))
    if n_matched_unique != n_matched:
        issues.append(f"{n_matched - n_matched_unique} duplicate canonical-key value(s) among matched manifest rows.")

    missing_in_manifest = set(prediction_values) - set(matched_keys)
    if missing_in_manifest:
        preview = sorted(missing_in_manifest)[:5]
        issues.append(f"{len(missing_in_manifest)} prediction identifier(s) not found in manifest.csv, e.g. {preview}")

    if n_pred != n_matched:
        issues.append(f"prediction row count ({n_pred}) != matched manifest row count ({n_matched}).")

    bad_split = [row for row in matched_rows if row.get("split") != "test"]
    if bad_split:
        issues.append(f"{len(bad_split)} matched manifest row(s) do not have split=='test'.")

    bad_sample = [row for row in matched_rows if row.get("sample_id") not in TEST_SAMPLES]
    if bad_sample:
        bad_ids = sorted({row.get("sample_id") for row in bad_sample})
        issues.append(f"{len(bad_sample)} matched manifest row(s) belong to sample_id(s) outside "
                       f"the frozen TEST_SAMPLES set: {bad_ids}")

    matched_sample_ids = {row.get("sample_id") for row in matched_rows}
    if matched_sample_ids != TEST_SAMPLES:
        issues.append(f"matched sample_id set {sorted(matched_sample_ids)} != expected frozen "
                       f"test set {sorted(TEST_SAMPLES)}.")

    return issues


def resolve_patch_path_like_full_run(raw_path, stage1_root):
    """Mirrors EXACTLY what the existing, unmodified load_raw_patch_stats() does before calling
    np.load(): it passes p['patch_path'] -- a value from the PREDICTIONS CSV, not manifest.csv --
    straight through with ZERO transformation. That call already succeeds in the full run
    ('raw patch stats: 16587 / 16587'), so the as-is string is tried first and is the expected
    match.

    A couple of narrow, clearly-labeled fallbacks are tried only if the as-is path does not
    resolve, in case a differently-shaped patch_path is ever encountered -- but nothing is ever
    silently accepted: the caller always learns exactly which strategy (if any) worked.
    Returns (resolved_path_or_None, strategy_description)."""
    p_as_is = Path(raw_path)
    if p_as_is.exists():
        return p_as_is, "as-is (identical to load_raw_patch_stats()'s np.load(p['patch_path']) -- no transformation)"

    normalized = raw_path.replace("\\", "/")
    p_norm = Path(normalized)
    if p_norm.exists():
        return p_norm, "as-is with '\\' normalized to '/' (still no path-base change)"

    p_under_stage1 = Path(stage1_root) / normalized
    if p_under_stage1.exists():
        return p_under_stage1, f"resolved relative to stage1 root ({stage1_root}) -- NOT what load_raw_patch_stats() does; only a fallback"

    return None, "not found: tried as-is, separator-normalized, and relative-to-stage1-root"


def dry_run_check_run08_patch_paths_exist(predictions, stage1_root):
    """Verifies every Run08 PREDICTION row's patch_path resolves to an existing file on disk --
    using predictions.csv's own patch_path values, the exact same ones load_raw_patch_stats()
    passes to np.load() with no modification. (The previous version of this check incorrectly
    used manifest.csv's patch_path column, which can use a different path convention than
    predictions.csv and is never what actually gets loaded during the full run.)
    Path.exists() only -- never np.load(), never reads array contents."""
    found, missing = 0, []
    strategy_counts = {}
    for p in predictions:
        pp = p.get("patch_path", "")
        if not pp:
            missing.append({"patch_path": "<empty>", "resolved": None})
            continue
        resolved, strategy = resolve_patch_path_like_full_run(pp, stage1_root)
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        if resolved is not None:
            found += 1
        else:
            if len(missing) < 10:
                missing.append({"patch_path": pp, "resolved": None})
    checked = len(predictions)
    return {"checked": checked, "found": found, "missing_count": checked - found,
            "missing_examples": missing[:10], "strategy_counts": strategy_counts}


def run_dry_run(stage1_root, run_dir, split_manifest_path, manifest_csv_path, audit_path,
                 predictions_csv_path, out_dir, out_json, out_txt, overwrite):
    print("=" * 70)
    print("DRY RUN: diagnose_cross_sample_shift.py")
    print("Validation only. Will NOT call np.load() on patch arrays. Will NOT run the full "
          "cross-sample / epsilon-amplification analysis.")
    print("=" * 70)

    # ---- 1. resolved paths ----
    print("\n[1] Resolved paths:")
    print(f"  stage1 root                 : {stage1_root}")
    print(f"  manifest CSV                : {manifest_csv_path}")
    print(f"  split manifest              : {split_manifest_path}")
    print(f"  sample distribution audit   : {audit_path}")
    print(f"  Run08 predictions CSV       : {predictions_csv_path}")
    print(f"  output directory            : {out_dir}")

    # ---- 2. required files exist ----
    print("\n[2] Required input files:")
    required = {
        "manifest.csv": manifest_csv_path,
        "split_manifest.csv": split_manifest_path,
        "test_predictions_resumable.csv": predictions_csv_path,
    }
    missing_required = []
    for label, p in required.items():
        exists = p.exists()
        print(f"  {'FOUND' if exists else 'MISSING':7s} {label:32s} -> {p}")
        if not exists:
            missing_required.append(p)
    audit_exists = audit_path.exists()
    print(f"  {'FOUND' if audit_exists else 'MISSING (optional)':19s} "
          f"{'sample_distribution_audit.json':32s} -> {audit_path}")
    if missing_required:
        raise SystemExit(f"BLOCKED: {len(missing_required)} required input file(s) missing "
                          f"(see [2] above). Cannot proceed even with --dry-run.")

    # ---- 3/4. load + validate split_manifest.csv, manifest.csv, predictions CSV; print columns ----
    print("\n[3/4] Loading and validating CSVs, inspecting actual column names...")
    split_fields = get_csv_fieldnames(split_manifest_path)
    manifest_fields = get_csv_fieldnames(manifest_csv_path)
    predictions_fields = get_csv_fieldnames(predictions_csv_path)
    print(f"  split_manifest.csv columns  : {split_fields}")
    print(f"  manifest.csv columns        : {manifest_fields}")
    print(f"  predictions CSV columns     : {predictions_fields}")
    for required_col in ("sample_id", "split"):
        if required_col not in split_fields:
            raise SystemExit(f"BLOCKED: split_manifest.csv is missing required column '{required_col}'.")
    for required_col in ("sample_id", "split", "label"):
        if required_col not in manifest_fields:
            raise SystemExit(f"BLOCKED: manifest.csv is missing required column '{required_col}'.")
    for required_col in ("sample_id", "label", "probability"):
        if required_col not in predictions_fields:
            raise SystemExit(f"BLOCKED: predictions CSV is missing required column '{required_col}'.")
    print("  All three CSVs contain their minimum required columns.")

    # ---- 5. frozen split: disjointness + print sample IDs per split (never recomputed) ----
    print("\n[5] Frozen split verification (loaded from split_manifest.csv, not recomputed):")
    split_assignment, by_split = dry_run_load_split_manifest(split_manifest_path)
    for s in SPLITS:
        print(f"  {s} ({len(by_split[s])} samples): {sorted(by_split[s])}")
    print("  train/val/test sample IDs confirmed disjoint (no leakage).")
    if by_split["test"] != TEST_SAMPLES:
        raise SystemExit(
            f"BLOCKED: split_manifest.csv's test samples {sorted(by_split['test'])} do not "
            f"match the expected frozen test set {sorted(TEST_SAMPLES)}."
        )
    print(f"  test split matches the expected frozen test set: {sorted(TEST_SAMPLES)}")

    # ---- 6. join-key detection + canonical matching representation (moved before count ----
    # ---- validation: the matched set below depends on it). This does NOT compare raw ----
    # ---- patch_path strings -- it reproduces the exact semantics of the existing, unmodified ----
    # ---- load_manifest_by_filename() + cross_validate_predictions_against_manifest(). ----
    print("\n[6] Predictions-to-manifest matching (inspecting actual headers, not assuming a representation):")
    join_key, reason = detect_join_key(manifest_fields, predictions_fields)
    print(f"  {reason}")
    if join_key is None:
        raise SystemExit(
            "BLOCKED: no shared join column found between manifest.csv and the predictions CSV. "
            "The full run's cross-validation step would be unable to proceed either -- stopping "
            "the dry-run here rather than reporting counts that can't be tied to the actual "
            "Run08 prediction set."
        )
    canonical = determine_canonical_matching(manifest_fields, predictions_fields)
    if canonical is None:
        raise SystemExit("BLOCKED: could not determine any usable matching representation between "
                          "manifest.csv and the predictions CSV.")
    print(f"  Join key selected: '{canonical['join_key_label']}'")
    print(f"  Canonical matching representation: {canonical['description']}")

    # ---- 7. full current manifest counts by split -- INFORMATIONAL ONLY ----
    print("\n[7] Full current manifest counts by split (informational -- NOT the Run08 "
          "evaluation set; manifest.csv may contain test-split rows added/changed after Run08 "
          "was evaluated):")
    full_manifest_counts = dry_run_manifest_counts_by_split(manifest_csv_path)
    for s in SPLITS:
        c = full_manifest_counts[s]
        print(f"  {s:5s}: positive={c['positive']:6d}  negative={c['negative']:6d}  total={c['total']:6d}")

    # ---- 8. build the Run08 prediction-linked matched manifest set (authoritative) ----
    print("\n[8] Building the Run08 prediction-linked matched manifest set "
          f"(predictions CSV is authoritative; matching representation: {canonical['description']})...")
    prediction_key_fn = canonical["prediction_key_fn"]
    manifest_key_fn = canonical["manifest_key_fn"]
    if not {"filename", "patch_path"}.issubset(predictions_fields):
        raise SystemExit(
            "BLOCKED: predictions CSV is missing 'filename' and/or 'patch_path' -- these are "
            "required by the existing, unmodified load_predictions() (used by the full run) and "
            "by this dry-run's patch-path existence check."
        )
    predictions = load_predictions(predictions_csv_path)  # SAME loader the full run uses
    prediction_values = [prediction_key_fn(row) for row in predictions]
    prediction_id_set = set(prediction_values)
    print(f"  predictions CSV rows: {len(prediction_values)}  (distinct prediction identifiers: {len(prediction_id_set)})")
    matched_rows = build_run08_matched_manifest_rows(manifest_csv_path, manifest_key_fn, prediction_id_set)
    print(f"  matched manifest rows: {len(matched_rows)}")

    integrity_issues = run08_matched_set_integrity_checks(
        prediction_values, matched_rows, manifest_key_fn, canonical["description"]
    )
    if integrity_issues:
        print(f"\n  BLOCKED: {len(integrity_issues)} integrity issue(s) found in the Run08 "
              f"prediction-linked matched set:")
        for issue in integrity_issues:
            print(f"    - {issue}")
        raise SystemExit("BLOCKED: Run08 prediction-linked matched-set integrity checks failed (see above).")
    print("  Integrity checks passed: every prediction identifier matches exactly one manifest "
          "row, no duplicates on either side, every matched row has split=='test', every "
          "matched sample_id is in the frozen TEST_SAMPLES set, and the matched sample_id set "
          "is exactly the frozen test set.")

    # ---- 9. Run08 prediction-linked count validation (authoritative) ----
    print("\n[9] Run08 prediction-linked manifest counts (authoritative -- computed ONLY from "
          "manifest rows matched to the predictions CSV):")
    run08_counts, run08_per_sample = compute_run08_matched_counts(matched_rows)
    print(f"  total={run08_counts['total']}  positive={run08_counts['positive']}  negative={run08_counts['negative']}")
    print(f"  expected: total={EXPECTED_TOTAL} positive={EXPECTED_POSITIVE} negative={EXPECTED_NEGATIVE}")
    count_mismatch = (run08_counts["total"] != EXPECTED_TOTAL
                       or run08_counts["positive"] != EXPECTED_POSITIVE
                       or run08_counts["negative"] != EXPECTED_NEGATIVE)
    print("  MISMATCH." if count_mismatch else "  MATCH.")

    print("\nRun08 prediction-linked per-sample test breakdown:")
    per_sample_mismatch = False
    for sid in sorted(TEST_SAMPLES):
        expected = EXPECTED_PER_SAMPLE_COUNTS[sid]
        actual = run08_per_sample.get(sid, {"total": 0, "positive": 0, "negative": 0})
        match = actual == expected
        per_sample_mismatch = per_sample_mismatch or not match
        tag = "MATCH" if match else "MISMATCH"
        print(f"  {sid} -> {actual['total']} patches, {actual['positive']} positive, "
              f"{actual['negative']} negative [{tag}]")
    if count_mismatch or per_sample_mismatch:
        print("\n  WARNING: count mismatch(es) detected in the Run08 prediction-linked set. "
              "The full run's own sanity checks would raise a hard error (BLOCKED) at this point.")

    # ---- 10. patch path existence, using PREDICTIONS' own patch_path (matches load_raw_patch_stats) ----
    print(f"\n[10] Run08 prediction-linked patch path existence check "
          f"({len(predictions)} paths, filesystem stat only, no np.load()):")
    path_check = dry_run_check_run08_patch_paths_exist(predictions, stage1_root)
    print(f"  checked={path_check['checked']}  found={path_check['found']}  "
          f"missing={path_check['missing_count']}")
    print(f"  resolution strategy breakdown: {path_check['strategy_counts']}")
    if path_check["missing_count"] > 0:
        print(f"  example unresolved paths (original -> attempted resolutions all failed): "
              f"{[m['patch_path'] for m in path_check['missing_examples']]}")
        print("  WARNING: the full run would fail when it attempts to np.load() a missing path.")
    else:
        print(f"  All {path_check['checked']} Run08 prediction-linked patch paths exist on disk.")

    # ---- 11. output directory overwrite protection ----
    print("\n[11] Output directory overwrite protection:")
    print(f"  output JSON : {out_json}  ({'EXISTS' if out_json.exists() else 'does not exist'})")
    print(f"  output TXT  : {out_txt}  ({'EXISTS' if out_txt.exists() else 'does not exist'})")
    if (out_json.exists() or out_txt.exists()) and not overwrite:
        print("  Full run (without --overwrite) would BLOCK here rather than overwrite existing output.")
    elif (out_json.exists() or out_txt.exists()) and overwrite:
        print("  --overwrite is set: full run would proceed and overwrite the existing output above.")
    else:
        print("  No existing output at this path; full run would write fresh files.")

    # ---- 12. what the full run will do next ----
    print("\n[12] Full analysis that will run after validation (not performed in --dry-run):")
    print("  - Load ALL predictions rows and cross-validate each against manifest.csv via "
          "load_manifest_by_filename() + cross_validate_predictions_against_manifest() "
          f"(matching representation: {canonical['description']}).")
    print("  - Recompute pooled + per-sample ROC-AUC/PR-AUC from test_predictions_resumable.csv "
          "and diff against the reported Run08 values.")
    print("  - Load sample_distribution_audit.json (if present) for train/val/test raw "
          "intensity context.")
    print(f"  - np.load() every one of the {path_check['checked']} Run08 prediction-linked test "
          "patch .npy arrays, compute each patch's raw (pre-normalization) mean/std, and run "
          "the epsilon-amplification analysis (per-sample, per-label raw-std distributions, "
          "epsilon-dominated fraction, and correlation between raw std/mean and predicted "
          "probability).")
    print(f"  - Write {out_json} and {out_txt}.")

    print("\nDRY RUN COMPLETE. No full patch analysis performed.")


# --------------------------------------------------------------------------
# Metrics recomputation (sanity check against the numbers you reported)
# --------------------------------------------------------------------------

def compute_roc_pr(y_true, probs):
    roc_auc = float(roc_auc_score(y_true, probs)) if len(set(y_true.tolist())) > 1 else float("nan")
    pr_auc = float(average_precision_score(y_true, probs))
    return roc_auc, pr_auc


def recompute_and_compare_metrics(predictions):
    y_true = np.array([p["label"] for p in predictions], dtype=np.float32)
    probs = np.array([p["probability"] for p in predictions], dtype=np.float64)
    overall_roc, overall_pr = compute_roc_pr(y_true, probs)

    mismatches = []
    if abs(overall_roc - REPORTED_OVERALL["roc_auc"]) > METRIC_MISMATCH_TOLERANCE:
        mismatches.append(f"overall ROC-AUC: recomputed={overall_roc:.6f} reported={REPORTED_OVERALL['roc_auc']:.6f}")
    if abs(overall_pr - REPORTED_OVERALL["pr_auc"]) > METRIC_MISMATCH_TOLERANCE:
        mismatches.append(f"overall PR-AUC: recomputed={overall_pr:.6f} reported={REPORTED_OVERALL['pr_auc']:.6f}")

    per_sample = {}
    by_sample = {sid: [] for sid in TEST_SAMPLES}
    for p in predictions:
        by_sample[p["sample_id"]].append(p)

    for sid in sorted(TEST_SAMPLES):
        rows = by_sample[sid]
        yt = np.array([r["label"] for r in rows], dtype=np.float32)
        pr = np.array([r["probability"] for r in rows], dtype=np.float64)
        roc_auc, pr_auc = compute_roc_pr(yt, pr)
        pos_mean = float(pr[yt == 1.0].mean()) if (yt == 1.0).any() else float("nan")
        neg_mean = float(pr[yt == 0.0].mean()) if (yt == 0.0).any() else float("nan")
        per_sample[sid] = {
            "n": len(rows), "n_positive": int((yt == 1.0).sum()), "n_negative": int((yt == 0.0).sum()),
            "roc_auc": roc_auc, "pr_auc": pr_auc, "positive_mean": pos_mean, "negative_mean": neg_mean,
        }
        rep = REPORTED_PER_SAMPLE.get(sid, {})
        if "roc_auc" in rep and abs(roc_auc - rep["roc_auc"]) > METRIC_MISMATCH_TOLERANCE:
            mismatches.append(f"{sid} ROC-AUC: recomputed={roc_auc:.6f} reported={rep['roc_auc']:.6f}")
        if "pr_auc" in rep and abs(pr_auc - rep["pr_auc"]) > METRIC_MISMATCH_TOLERANCE:
            mismatches.append(f"{sid} PR-AUC: recomputed={pr_auc:.6f} reported={rep['pr_auc']:.6f}")

    return {"overall": {"roc_auc": overall_roc, "pr_auc": overall_pr}, "per_sample": per_sample}, mismatches


# --------------------------------------------------------------------------
# Raw (pre-normalization) patch statistics -- the epsilon-amplification hypothesis test
# --------------------------------------------------------------------------

def compute_predictions_fingerprint(predictions):
    """Deterministic fingerprint tying a checkpoint to ONE specific, ordered prediction set.
    Computed from the ordered sequence of each row's own 'filename' (the same canonical
    identifier the checkpoint is keyed by) plus the total row count. If the predictions CSV is
    replaced, reordered, or has rows added/removed, this fingerprint changes and any existing
    checkpoint is treated as stale rather than silently reused against the wrong data."""
    hasher = hashlib.sha256()
    hasher.update(str(len(predictions)).encode("utf-8"))
    for p in predictions:
        hasher.update(b"\n")
        hasher.update(p["filename"].encode("utf-8"))
    return hasher.hexdigest()


def _atomic_write_bytes(path, write_fn):
    """Writes to a temp file in the same directory, then os.replace()s it into place, so a
    Ctrl+C or crash mid-write can never leave a half-written checkpoint file at `path`."""
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            write_fn(f)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def save_raw_stats_checkpoint(checkpoint_npz_path, checkpoint_meta_path, filenames, means, stds, fingerprint, total_expected):
    """Atomically persists the completed-so-far (filename, mean, std) triples plus the
    fingerprint tying them to a specific prediction set. Safe to call repeatedly (e.g. every
    RAW_STATS_CHECKPOINT_EVERY patches, and once more on KeyboardInterrupt)."""
    filenames_arr = np.array(filenames, dtype=object)
    means_arr = np.array(means, dtype=np.float64)
    stds_arr = np.array(stds, dtype=np.float64)

    def _write_npz(f):
        np.savez(f, filenames=filenames_arr, means=means_arr, stds=stds_arr)
    _atomic_write_bytes(checkpoint_npz_path, _write_npz)

    meta = {
        "prediction_set_fingerprint": fingerprint,
        "total_expected": total_expected,
        "n_completed": len(filenames),
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
    }

    def _write_meta(f):
        f.write(json.dumps(meta, indent=2).encode("utf-8"))
    _atomic_write_bytes(checkpoint_meta_path, _write_meta)


def load_raw_stats_checkpoint(checkpoint_npz_path, checkpoint_meta_path, expected_fingerprint, total_expected):
    """Loads and validates an existing checkpoint. Returns a dict {filename: (mean, std)} of
    already-completed patches, or an EMPTY dict if no valid checkpoint exists to resume from.

    Validation is strict and never silently accepts questionable cached data:
      - if only one of the two checkpoint files exists, that's corruption -> hard error
      - if the stored fingerprint doesn't match the CURRENT prediction set -> hard error
        (never silently reused against a different/reordered/changed prediction set)
      - if the npz arrays have mismatched lengths, or contain duplicate filenames -> hard error
      - n_completed in the meta file must match the actual array length -> hard error otherwise
    In every hard-error case, the message tells the user to inspect or pass
    --reset-raw-stats-cache; nothing is auto-deleted on their behalf."""
    npz_exists = Path(checkpoint_npz_path).exists()
    meta_exists = Path(checkpoint_meta_path).exists()
    if not npz_exists and not meta_exists:
        return {}
    if npz_exists != meta_exists:
        raise ValueError(
            f"BLOCKED: raw-patch-stats checkpoint is corrupted (only one of the two checkpoint "
            f"files exists: npz_exists={npz_exists} meta_exists={meta_exists}). Refusing to "
            f"resume from a partial/corrupted checkpoint. Pass --reset-raw-stats-cache to "
            f"discard it and start fresh, or restore/remove the files manually:\n"
            f"  {checkpoint_npz_path}\n  {checkpoint_meta_path}"
        )

    try:
        meta = json.loads(Path(checkpoint_meta_path).read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"BLOCKED: could not parse checkpoint metadata at {checkpoint_meta_path}: {e}. "
                          f"Pass --reset-raw-stats-cache to discard it and start fresh.")

    if meta.get("prediction_set_fingerprint") != expected_fingerprint:
        raise ValueError(
            "BLOCKED: existing raw-patch-stats checkpoint does not match the CURRENT Run08 "
            "prediction set (fingerprint mismatch -- the predictions CSV appears to have "
            "changed, been reordered, or this checkpoint belongs to a different run). Refusing "
            "to silently reuse it. Pass --reset-raw-stats-cache to discard the stale checkpoint "
            "and start fresh."
        )
    if meta.get("total_expected") != total_expected:
        raise ValueError(
            f"BLOCKED: checkpoint's total_expected ({meta.get('total_expected')}) does not match "
            f"the current prediction row count ({total_expected}). Refusing to resume from a "
            f"mismatched checkpoint. Pass --reset-raw-stats-cache to discard it and start fresh."
        )

    try:
        with np.load(checkpoint_npz_path, allow_pickle=True) as npz:
            filenames = list(npz["filenames"])
            means = npz["means"]
            stds = npz["stds"]
    except (OSError, ValueError, KeyError) as e:
        raise ValueError(f"BLOCKED: could not read checkpoint npz at {checkpoint_npz_path}: {e}. "
                          f"Pass --reset-raw-stats-cache to discard it and start fresh.")

    if not (len(filenames) == len(means) == len(stds)):
        raise ValueError(
            f"BLOCKED: corrupted checkpoint -- array length mismatch (filenames={len(filenames)}, "
            f"means={len(means)}, stds={len(stds)}). Pass --reset-raw-stats-cache to discard it."
        )
    if meta.get("n_completed") != len(filenames):
        raise ValueError(
            f"BLOCKED: corrupted checkpoint -- meta reports n_completed={meta.get('n_completed')} "
            f"but the npz contains {len(filenames)} entries. Pass --reset-raw-stats-cache to "
            f"discard it."
        )
    if len(set(filenames)) != len(filenames):
        n_dupes = len(filenames) - len(set(filenames))
        raise ValueError(
            f"BLOCKED: corrupted checkpoint -- {n_dupes} duplicate filename(s) found among "
            f"cached entries. Refusing to use ambiguous cached values. Pass "
            f"--reset-raw-stats-cache to discard it."
        )

    return {fn: (float(m), float(s)) for fn, m, s in zip(filenames, means, stds)}


def load_raw_patch_stats(predictions, patch_size, out_dir, progress_every=2000,
                          checkpoint_every=RAW_STATS_CHECKPOINT_EVERY):
    """For every prediction row, loads the RAW .npy patch (before any normalization) and
    computes its own mean/std -- i.e. exactly the statistic Run08's instance normalization
    would have divided by. Read-only with respect to every input file: this never writes to
    patch_path, never re-derives labels, and is independent of the model/checkpoint entirely.

    RESUMABLE: progress is checkpointed to
        <out_dir>/raw_patch_stats_checkpoint.npz
        <out_dir>/raw_patch_stats_checkpoint.meta.json
    every `checkpoint_every` newly-computed patches, and immediately on KeyboardInterrupt. On
    the next invocation with the SAME prediction set (verified via a content fingerprint -- see
    compute_predictions_fingerprint), already-completed patches are loaded from the checkpoint
    and skipped, so an interrupted run resumes rather than restarting from patch 0.

    The checkpoint is deliberately KEPT after a successful full run (not deleted) -- it is a
    reusable, validated cache; re-running this script again with the same prediction set will
    load it back and do zero additional np.load() calls. Use --reset-raw-stats-cache to discard
    it explicitly if a clean re-computation is ever desired.

    Regardless of what order patches were computed or loaded from cache in, the returned
    raw_means/raw_stds arrays are reconstructed by iterating `predictions` in its ORIGINAL
    order and looking up each row's own filename -- never by iterating any dict/cache in
    whatever order it happens to hold entries.
    """
    n = len(predictions)
    filenames_in_order = [p["filename"] for p in predictions]
    if len(set(filenames_in_order)) != n:
        n_dupes = n - len(set(filenames_in_order))
        raise ValueError(
            f"BLOCKED: {n_dupes} duplicate 'filename' value(s) found among the {n} prediction "
            f"rows. The raw-patch-stats checkpoint is keyed by filename and requires uniqueness "
            f"-- refusing to proceed with an ambiguous prediction set."
        )

    fingerprint = compute_predictions_fingerprint(predictions)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_npz_path = out_dir / RAW_STATS_CHECKPOINT_FILENAME
    checkpoint_meta_path = out_dir / RAW_STATS_CHECKPOINT_META_FILENAME

    cache = load_raw_stats_checkpoint(checkpoint_npz_path, checkpoint_meta_path, fingerprint, n)
    n_cached_at_start = len(cache)
    if n_cached_at_start > 0:
        print(f"  Found checkpoint: {n_cached_at_start} / {n} patches already processed.")
        print(f"  Resuming raw patch statistics...")

    since_last_save = 0

    def _flush():
        save_raw_stats_checkpoint(checkpoint_npz_path, checkpoint_meta_path,
                                   list(cache.keys()), [v[0] for v in cache.values()],
                                   [v[1] for v in cache.values()], fingerprint, n)

    try:
        for i, p in enumerate(predictions):
            filename = p["filename"]
            if filename in cache:
                continue  # already computed in a previous (interrupted) run
            arr = np.load(p["patch_path"])
            if arr.shape != (patch_size, patch_size):
                raise ValueError(f"SHAPE MISMATCH: {p['patch_path']} has shape {arr.shape}, expected ({patch_size}, {patch_size}).")
            arr64 = arr.astype(np.float64)
            cache[filename] = (float(arr64.mean()), float(arr64.std()))
            since_last_save += 1

            if progress_every and ((len(cache)) % progress_every == 0 or len(cache) == n):
                print(f"  raw patch stats: {len(cache)} / {n}")
            if checkpoint_every and since_last_save >= checkpoint_every:
                _flush()
                since_last_save = 0
    except KeyboardInterrupt:
        _flush()
        print(f"\nINTERRUPTED: saved checkpoint with {len(cache)} / {n} completed patches.")
        print("Run the same command again to resume.")
        raise SystemExit(130)  # clean exit, no traceback; 130 = conventional SIGINT exit code

    # final checkpoint save (covers the tail end since the last periodic flush)
    _flush()

    # ---- integrity: every prediction row has exactly one cached result, no more, no fewer ----
    if len(cache) != n:
        raise ValueError(f"BLOCKED: expected {n} completed patches after the loop, found {len(cache)} in cache.")
    missing = [fn for fn in filenames_in_order if fn not in cache]
    if missing:
        raise ValueError(f"BLOCKED: {len(missing)} prediction row(s) have no cached result after "
                          f"processing, e.g. {missing[:5]}.")

    # ---- reconstruct output arrays strictly in `predictions`' original order (never dict order) ----
    raw_means = np.array([cache[fn][0] for fn in filenames_in_order], dtype=np.float64)
    raw_stds = np.array([cache[fn][1] for fn in filenames_in_order], dtype=np.float64)
    return raw_means, raw_stds


def percentile_block(arr):
    if len(arr) == 0:
        return {p: float("nan") for p in PERCENTILES}
    return {p: float(np.percentile(arr, p)) for p in PERCENTILES}


def pearson_or_nan(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def analyze_epsilon_amplification(predictions, raw_means, raw_stds, norm_eps, epsilon_dominated_percentile):
    """Per-sample, per-label breakdown of raw std/mean vs predicted probability, plus an
    'epsilon-dominated' flag for patches whose raw std sits in the lowest tail of the WHOLE test
    set (i.e. patches where std + eps is proportionally most inflated by eps)."""
    probs = np.array([p["probability"] for p in predictions], dtype=np.float64)
    labels = np.array([p["label"] for p in predictions], dtype=np.float32)
    sample_ids = np.array([p["sample_id"] for p in predictions])

    global_std_threshold = float(np.percentile(raw_stds, epsilon_dominated_percentile))
    epsilon_dominated_mask = raw_stds <= global_std_threshold
    eps_fraction_relative_to_std = norm_eps / (raw_stds + norm_eps)

    result = {
        "norm_eps": norm_eps,
        "global_raw_std_percentiles": percentile_block(raw_stds),
        "epsilon_dominated_definition": (
            f"raw patch std at or below the {epsilon_dominated_percentile}th percentile of "
            f"ALL test-set raw stds (i.e. the flattest {epsilon_dominated_percentile}% of test "
            f"patches by raw contrast)"
        ),
        "epsilon_dominated_global_std_threshold": global_std_threshold,
        "per_sample": {},
    }

    for sid in sorted(TEST_SAMPLES):
        smask = sample_ids == sid
        for label_name, label_val in [("positive", 1.0), ("negative", 0.0)]:
            lmask = smask & (labels == label_val)
            n = int(lmask.sum())
            if n == 0:
                continue
            entry = {
                "n": n,
                "raw_std_percentiles": percentile_block(raw_stds[lmask]),
                "raw_mean_percentiles": percentile_block(raw_means[lmask]),
                "mean_eps_fraction_of_denominator": float(eps_fraction_relative_to_std[lmask].mean()),
                "fraction_epsilon_dominated": float(epsilon_dominated_mask[lmask].mean()),
                "prob_mean": float(probs[lmask].mean()),
                "prob_median": float(np.median(probs[lmask])),
                "corr_raw_std_vs_prob": pearson_or_nan(raw_stds[lmask], probs[lmask]),
                "corr_raw_mean_vs_prob": pearson_or_nan(raw_means[lmask], probs[lmask]),
            }
            result["per_sample"].setdefault(sid, {})[label_name] = entry

    return result


# --------------------------------------------------------------------------
# Cross-sample raw intensity context (uses sample_distribution_audit.json as read-only input)
# --------------------------------------------------------------------------

def load_sample_distribution_audit(path):
    if not path.exists():
        print(f"NOTE: {path} not found -- skipping cross-sample audit-file comparison "
              f"(this section is supplementary; the epsilon-amplification analysis above does "
              f"not depend on it).")
        return None
    return json.loads(path.read_text())


def summarize_audit_context(audit, split_assignment):
    """Read-only summary: for every sample_id in the audit file, report its (mean, std) and
    which split it belongs to (from split_manifest.csv), so 44b6_0b24845f's raw profile can be
    compared against the OTHER test samples and against the train distribution without
    recomputing anything."""
    if audit is None:
        return None
    entries = audit if isinstance(audit, dict) and "samples" not in audit else audit.get("samples", audit)
    rows = []
    for sid, stats in entries.items():
        if not isinstance(stats, dict) or "mean" not in stats:
            continue
        rows.append({
            "sample_id": sid, "split": split_assignment.get(sid, "<unknown>"),
            "mean": stats.get("mean"), "std": stats.get("std"),
        })
    return rows


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-root", default=STAGE1_ROOT_DEFAULT)
    parser.add_argument("--run-dir", default=RUN_DIR_DEFAULT)
    parser.add_argument("--split-manifest", default=None, help="Default: <stage1-root>/split_manifest.csv")
    parser.add_argument("--manifest-csv", default=None, help="Default: <stage1-root>/manifest.csv")
    parser.add_argument("--sample-distribution-audit", default=None, help="Default: <stage1-root>/sample_distribution_audit.json")
    parser.add_argument("--predictions-csv", default=None, help="Default: <run-dir>/test_predictions_resumable.csv")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--norm-eps", type=float, default=1e-6, help="Must match train_exp05_run08.py's NORM_EPS")
    parser.add_argument("--epsilon-dominated-percentile", type=float, default=5.0)
    parser.add_argument("--out-dir", default=None, help="Default: <run-dir>/diagnostics")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reset-raw-stats-cache", action="store_true",
                         help="Delete ONLY the raw-patch-stats checkpoint "
                              f"({RAW_STATS_CHECKPOINT_FILENAME} / {RAW_STATS_CHECKPOINT_META_FILENAME}) "
                              "in the output directory before running, then proceed normally. "
                              "Does not touch predictions, manifest, split, or any prior "
                              "diagnostic JSON/TXT output.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Validation only: resolve paths, check files exist, inspect CSV "
                              "columns, verify the frozen split and manifest counts, check test "
                              "patch_path existence, and report overwrite status. Does NOT call "
                              "np.load() on any patch and does NOT run the full cross-sample / "
                              "epsilon-amplification analysis.")
    args = parser.parse_args()

    stage1_root = Path(args.stage1_root)
    run_dir = Path(args.run_dir)
    split_manifest_path = Path(args.split_manifest) if args.split_manifest else stage1_root / "split_manifest.csv"
    manifest_csv_path = Path(args.manifest_csv) if args.manifest_csv else stage1_root / "manifest.csv"
    audit_path = Path(args.sample_distribution_audit) if args.sample_distribution_audit else stage1_root / "sample_distribution_audit.json"
    predictions_csv_path = Path(args.predictions_csv) if args.predictions_csv else run_dir / "test_predictions_resumable.csv"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "diagnostics"
    out_json = out_dir / "cross_sample_shift_diagnostic.json"
    out_txt = out_dir / "cross_sample_shift_diagnostic.txt"

    if args.reset_raw_stats_cache:
        checkpoint_npz_path = out_dir / RAW_STATS_CHECKPOINT_FILENAME
        checkpoint_meta_path = out_dir / RAW_STATS_CHECKPOINT_META_FILENAME
        removed_any = False
        for p in [checkpoint_npz_path, checkpoint_meta_path]:
            if p.exists():
                p.unlink()
                print(f"--reset-raw-stats-cache: removed {p}")
                removed_any = True
        if not removed_any:
            print("--reset-raw-stats-cache: no existing raw-patch-stats checkpoint found; nothing to remove.")

    if args.dry_run:
        run_dry_run(stage1_root, run_dir, split_manifest_path, manifest_csv_path, audit_path,
                    predictions_csv_path, out_dir, out_json, out_txt, args.overwrite)
        return

    for required in [split_manifest_path, manifest_csv_path, predictions_csv_path]:
        if not required.exists():
            raise SystemExit(f"BLOCKED: required file does not exist: {required}")

    if not args.overwrite:
        for p in [out_json, out_txt]:
            if p.exists():
                raise SystemExit(f"BLOCKED: {p} already exists. Pass --overwrite.")

    print("=" * 70)
    print("CROSS-SAMPLE SHIFT DIAGNOSTIC (Run08 post-hoc, diagnostic-only, no training)")
    print("=" * 70)

    # ---- 1. frozen split (loaded, never recomputed) ----
    split_assignment = load_split_manifest(split_manifest_path)
    test_sample_ids_in_split = {sid for sid, sp in split_assignment.items() if sp == "test"}
    if test_sample_ids_in_split != TEST_SAMPLES:
        raise SystemExit(
            f"BLOCKED: split_manifest.csv's test samples {sorted(test_sample_ids_in_split)} do "
            f"not match the expected frozen test set {sorted(TEST_SAMPLES)}. Refusing to "
            f"diagnose against an unexpected split."
        )
    print(f"\nFrozen split loaded and verified: test = {sorted(TEST_SAMPLES)} (unchanged).")

    # ---- 2/3. load + cross-validate predictions, recompute metrics ----
    manifest_by_filename = load_manifest_by_filename(manifest_csv_path)
    predictions = load_predictions(predictions_csv_path)
    print(f"\nLoaded {len(predictions)} prediction rows from {predictions_csv_path}")
    cross_validate_predictions_against_manifest(predictions, manifest_by_filename, split_assignment)

    recomputed, mismatches = recompute_and_compare_metrics(predictions)
    print(f"\nRecomputed overall: ROC-AUC={recomputed['overall']['roc_auc']:.6f}  "
          f"PR-AUC={recomputed['overall']['pr_auc']:.6f}")
    for sid in sorted(TEST_SAMPLES):
        m = recomputed["per_sample"][sid]
        print(f"  {sid}: n={m['n']} pos={m['n_positive']} neg={m['n_negative']} "
              f"ROC-AUC={m['roc_auc']:.6f} PR-AUC={m['pr_auc']:.6f} "
              f"pos_mean={m['positive_mean']:.6f} neg_mean={m['negative_mean']:.6f}")
    if mismatches:
        print(f"\nWARNING: {len(mismatches)} recomputed metric(s) disagree with the reported "
              f"values by more than {METRIC_MISMATCH_TOLERANCE}:")
        for m in mismatches:
            print(f"  {m}")
    else:
        print(f"\nAll recomputed metrics match the reported Run08 values within "
              f"{METRIC_MISMATCH_TOLERANCE}. Proceeding on a verified predictions file.")

    # ---- 4. cross-sample raw intensity context (supplementary, read-only) ----
    audit = load_sample_distribution_audit(audit_path)
    audit_context = summarize_audit_context(audit, split_assignment)
    if audit_context:
        print(f"\nSample-level raw intensity audit (from {audit_path}, read-only):")
        for row in sorted(audit_context, key=lambda r: r["split"]):
            print(f"  [{row['split']:5s}] {row['sample_id']}: mean={row['mean']:.3f} std={row['std']:.3f}")

    # ---- 5. epsilon-amplification hypothesis test on RAW (pre-normalization) patches ----
    print(f"\nLoading raw (pre-normalization) patch arrays for all {len(predictions)} test "
          f"patches to compute each patch's own raw mean/std (the statistic instance "
          f"normalization divides by)...")
    print(f"  (resumable: progress checkpointed every {RAW_STATS_CHECKPOINT_EVERY} patches to "
          f"{out_dir / RAW_STATS_CHECKPOINT_FILENAME}; Ctrl+C saves immediately and exits cleanly)")
    raw_means, raw_stds = load_raw_patch_stats(predictions, args.patch_size, out_dir)
    eps_analysis = analyze_epsilon_amplification(
        predictions, raw_means, raw_stds, args.norm_eps, args.epsilon_dominated_percentile
    )

    print(f"\n--- Epsilon-amplification analysis (eps={args.norm_eps}) ---")
    print(f"Global raw-std percentiles across all {len(predictions)} test patches: "
          f"{eps_analysis['global_raw_std_percentiles']}")
    for sid in sorted(TEST_SAMPLES):
        print(f"\nSAMPLE: {sid}")
        for label_name in ["positive", "negative"]:
            entry = eps_analysis["per_sample"].get(sid, {}).get(label_name)
            if entry is None:
                print(f"  {label_name}: no patches")
                continue
            print(f"  {label_name} (n={entry['n']}): "
                  f"raw_std median={entry['raw_std_percentiles'][50]:.4f}  "
                  f"raw_mean median={entry['raw_mean_percentiles'][50]:.4f}  "
                  f"frac_epsilon_dominated={entry['fraction_epsilon_dominated']:.3f}  "
                  f"prob_mean={entry['prob_mean']:.6f}  "
                  f"corr(raw_std,prob)={entry['corr_raw_std_vs_prob']:.4f}  "
                  f"corr(raw_mean,prob)={entry['corr_raw_mean_vs_prob']:.4f}")

    # ---- assemble + save ----
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostic = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "trained_no_model": True,
        "predictions_csv_path": str(predictions_csv_path),
        "split_manifest_path": str(split_manifest_path),
        "manifest_csv_path": str(manifest_csv_path),
        "sample_distribution_audit_path": str(audit_path) if audit is not None else None,
        "recomputed_metrics": recomputed,
        "reported_metrics_for_comparison": {"overall": REPORTED_OVERALL, "per_sample": REPORTED_PER_SAMPLE},
        "metric_mismatches": mismatches,
        "sample_distribution_audit_context": audit_context,
        "epsilon_amplification_analysis": eps_analysis,
    }
    out_json.write_text(json.dumps(diagnostic, indent=2))

    txt_lines = [
        "=" * 70, "CROSS-SAMPLE SHIFT DIAGNOSTIC (Run08, diagnostic-only)", "=" * 70,
        f"\nRecomputed overall ROC-AUC={recomputed['overall']['roc_auc']:.6f}  "
        f"PR-AUC={recomputed['overall']['pr_auc']:.6f}",
    ]
    if mismatches:
        txt_lines.append(f"\nMETRIC MISMATCHES vs reported values ({len(mismatches)}):")
        txt_lines += [f"  {m}" for m in mismatches]
    else:
        txt_lines.append("\nRecomputed metrics match reported values.")
    txt_lines.append("\nPER-SAMPLE:")
    for sid in sorted(TEST_SAMPLES):
        m = recomputed["per_sample"][sid]
        txt_lines.append(f"  {sid}: n={m['n']} ROC-AUC={m['roc_auc']:.6f} PR-AUC={m['pr_auc']:.6f} "
                          f"pos_mean={m['positive_mean']:.6f} neg_mean={m['negative_mean']:.6f}")
    txt_lines.append("\nEPSILON-AMPLIFICATION ANALYSIS:")
    for sid in sorted(TEST_SAMPLES):
        for label_name in ["positive", "negative"]:
            entry = eps_analysis["per_sample"].get(sid, {}).get(label_name)
            if entry is None:
                continue
            txt_lines.append(
                f"  {sid} / {label_name}: n={entry['n']} raw_std_median="
                f"{entry['raw_std_percentiles'][50]:.4f} frac_epsilon_dominated="
                f"{entry['fraction_epsilon_dominated']:.3f} prob_mean={entry['prob_mean']:.6f} "
                f"corr(raw_std,prob)={entry['corr_raw_std_vs_prob']:.4f}"
            )
    out_txt.write_text("\n".join(txt_lines))

    print(f"\nDIAGNOSTIC SAVED:\n  {out_json}\n  {out_txt}")
    print("\nDIAGNOSTIC COMPLETE (no model trained, no existing file modified).")


if __name__ == "__main__":
    main()