"""
Experiment #5 geometry audit: independent GT-coordinate extraction.

READ-ONLY. Does not modify src/dataset.py, build_exp05_training_dataset.py, adaptive_detector.py,
manifest.csv, split_manifest.csv, or any .geff/.zarr file. Does not import torch. Does not train
or evaluate anything. Its only effect is writing one new CSV to the path you pass via --out-csv.

Reuses src.dataset.BioHubDataset UNMODIFIED (same class build_exp05_training_dataset.py uses),
and duplicates that script's own load_ground_truth_points() function VERBATIM (same isolation
rationale stated in that script's docstring: every audit/experiment script re-derives GT loading
independently rather than importing across experiment scripts) -- so this extraction goes
through the exact same code path that produced the real labels, not a re-implementation of it.

Per your requirements:
  - GT source: ONLY nodes/props/{t,y,x}/values from each sample's .geff graph (via
    BioHubDataset.load_graph), never the detector, never manifest.csv's candidate/label columns.
  - manifest.csv is not read at all. split_manifest.csv IS read, but ONLY for its sample_id
    column (to know which 20 samples to extract) -- it is never a source of coordinates.
  - Every GT point (one row per graph node) is written, not deduplicated or grouped by frame.

Usage (run from the LabOS project root, in your existing venv):

    # Step 1 -- report only, does NOT write gt_coordinates.csv yet
    python scripts/benchmark_v2/extract_gt_coordinates.py `
        --dataset-path D:\\Datasets\\BioHub\\biohub-cell-tracking-during-development `
        --stage1-dir results\\exp05_training_dataset\\run02_20samples `
        --report-only

    # Step 2 -- after reviewing the report, actually write the CSV
    python scripts/benchmark_v2/extract_gt_coordinates.py `
        --dataset-path D:\\Datasets\\BioHub\\biohub-cell-tracking-during-development `
        --stage1-dir results\\exp05_training_dataset\\run02_20samples `
        --out-csv results\\exp05_training_dataset\\run02_20samples\\gt_coordinates.csv
"""

import argparse
import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.dataset import BioHubDataset  # unmodified


# --------------------------------------------------------------------------
# VERBATIM copy of build_exp05_training_dataset.py::load_ground_truth_points, so this script
# goes through the identical GT-reading code path used to build the real labels. If you change
# that function in build_exp05_training_dataset.py, update this copy to match -- do not import
# across the two scripts (same isolation rationale that script's own docstring states for every
# Experiment #3/#4/#5 audit script).
# --------------------------------------------------------------------------

def load_ground_truth_points(dataset, sample):
    """Confirmed schema (nodes/props/<name>/values). Returns {frame_idx: (N,2) array of (y,x)},
    i.e. grouped by frame -- used here only to reproduce the exact per-frame structure
    build_exp05_training_dataset.py itself relies on for its length/consistency check."""
    graph = dataset.load_graph(sample)
    props = graph["nodes"]["props"]
    ts = np.asarray(props["t"]["values"][:])
    ys = np.asarray(props["y"]["values"][:])
    xs = np.asarray(props["x"]["values"][:])

    n_ids = graph["nodes"]["ids"].shape[0]
    if not (len(ts) == len(ys) == len(xs) == n_ids):
        raise ValueError(
            f"Sample {sample}: nodes/props array lengths (t={len(ts)}, y={len(ys)}, "
            f"x={len(xs)}) don't match nodes/ids length ({n_ids}) -- aborting rather than "
            f"extracting untrustworthy ground truth."
        )
    return ts, ys, xs  # RAW, ungrouped -- one entry per graph node, in original node order


def load_sample_ids_from_split_manifest(split_manifest_path):
    """ONLY source of the 20 sample_ids. Reads the sample_id column ONLY -- never a coordinate,
    never a label, never a candidate. Satisfies 'do NOT use manifest.csv as the source of GT
    coordinates except to identify the 20 sample IDs' by using split_manifest.csv instead, which
    structurally cannot contain GT/candidate coordinates at all (it is sample_id,split only)."""
    ids = []
    with open(split_manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        if "sample_id" not in reader.fieldnames:
            raise ValueError(
                f"{split_manifest_path} has columns {reader.fieldnames}, expected a "
                f"'sample_id' column."
            )
        for row in reader:
            ids.append(row["sample_id"])
    if len(set(ids)) != len(ids):
        raise ValueError(f"{split_manifest_path} contains duplicate sample_id rows.")
    return sorted(set(ids))


def find_expected_gt_count(dataset_summary_path):
    """Best-effort parse of dataset_summary.txt's 'Total GT points = N' line, for the
    expected-vs-extracted comparison you asked for. Purely informational -- never used to
    filter or alter the extraction itself."""
    if not dataset_summary_path.exists():
        return None
    text = dataset_summary_path.read_text()
    m = re.search(r"Total GT points\s*=\s*(\d+)", text)
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True,
                         help=r"e.g. D:\Datasets\BioHub\biohub-cell-tracking-during-development")
    parser.add_argument("--stage1-dir", required=True,
                         help=r"e.g. results\exp05_training_dataset\run02_20samples")
    parser.add_argument("--split-manifest", default=None,
                         help="Default: <stage1-dir>/split_manifest.csv")
    parser.add_argument("--dataset-summary", default=None,
                         help="Default: <stage1-dir>/dataset_summary.txt (informational only)")
    parser.add_argument("--out-csv", default=None,
                         help="Where to write gt_coordinates.csv. If omitted, this is a "
                              "report-only dry run and NO CSV is written.")
    parser.add_argument("--report-only", action="store_true",
                         help="Explicit alias for omitting --out-csv -- makes the dry-run "
                              "intent unambiguous in your shell history.")
    args = parser.parse_args()

    stage1_dir = Path(args.stage1_dir)
    split_manifest_path = Path(args.split_manifest) if args.split_manifest else stage1_dir / "split_manifest.csv"
    dataset_summary_path = Path(args.dataset_summary) if args.dataset_summary else stage1_dir / "dataset_summary.txt"

    if not split_manifest_path.exists():
        raise SystemExit(f"BLOCKED: {split_manifest_path} not found -- cannot identify the 20 "
                          f"sample IDs. Pass --split-manifest explicitly if it lives elsewhere.")

    dataset = BioHubDataset(args.dataset_path)  # unmodified constructor, read-only zarr access

    sample_ids = load_sample_ids_from_split_manifest(split_manifest_path)
    print(f"Sample IDs found in {split_manifest_path} ({len(sample_ids)} total):")
    for sid in sample_ids:
        print(f"  {sid}")
    if len(sample_ids) != 20:
        print(f"\nWARNING: expected 20 sample IDs per the task description, found "
              f"{len(sample_ids)}. Continuing anyway -- extracting whatever is listed.")

    print(f"\nLocating .geff files under {dataset.train_dir} ...")
    geff_report = []
    for sid in sample_ids:
        p = dataset.train_dir / f"{sid}.geff"
        geff_report.append((sid, p, p.exists()))
        status = "FOUND" if p.exists() else "*** MISSING ***"
        size = f"{p.stat().st_size:,} bytes" if p.exists() and p.is_file() else (
            "directory (zarr group)" if p.exists() else "n/a")
        print(f"  {sid}: {p}  [{status}]  {size}")

    missing = [sid for sid, p, exists in geff_report if not exists]
    if missing:
        raise SystemExit(f"\nBLOCKED: {len(missing)} .geff path(s) not found: {missing}. "
                          f"Stopping before reading any graph.")

    print(f"\nRead path used for every sample (from build_exp05_training_dataset.py's own "
          f"load_ground_truth_points, reused verbatim):")
    print(f"  graph = dataset.load_graph(sample_id)   "
          f"# zarr.open_group(train_dir / f'{{sample_id}}.geff', mode='r')")
    print(f"  ts = graph['nodes']['props']['t']['values'][:]")
    print(f"  ys = graph['nodes']['props']['y']['values'][:]")
    print(f"  xs = graph['nodes']['props']['x']['values'][:]")
    print(f"  (length-checked against graph['nodes']['ids'].shape[0] before use)")

    expected_total = find_expected_gt_count(dataset_summary_path)
    print(f"\nExpected total GT points from {dataset_summary_path}: "
          f"{expected_total if expected_total is not None else 'NOT FOUND / not parseable'}")

    print("\nExtracting...")
    all_rows = []
    per_sample_counts = {}
    for sid in sample_ids:
        ts, ys, xs = load_ground_truth_points(dataset, sid)
        n = len(ts)
        per_sample_counts[sid] = n
        for t, y, x in zip(ts, ys, xs):
            all_rows.append((sid, int(t), float(y), float(x)))
        print(f"  {sid}: {n} GT points (frames spanned: {int(np.min(ts))}-{int(np.max(ts))})" if n else f"  {sid}: 0 GT points")

    total_extracted = len(all_rows)
    print(f"\nTotal GT points extracted: {total_extracted}")
    if expected_total is not None:
        diff = total_extracted - expected_total
        print(f"Expected (dataset_summary.txt): {expected_total}")
        print(f"Difference (extracted - expected): {diff:+d}")
        if diff != 0:
            print("NOTE: a non-zero difference is expected if dataset_summary.txt's 'Total GT "
                  "points' was accumulated only over frames that had >=1 detector candidate "
                  "(build_exp05_training_dataset.py's own counts['total_gt'] is incremented "
                  "inside the per-frame loop regardless of candidate count, so it SHOULD match "
                  "-- investigate any mismatch rather than assuming it's benign).")

    print("\nPer-sample GT counts:")
    for sid in sample_ids:
        print(f"  {sid}: {per_sample_counts[sid]}")

    # ---- missing / duplicate checks ----
    n_missing_vals = sum(1 for _, t, y, x in all_rows if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (t, y, x)))
    seen = {}
    n_exact_dup_rows = 0
    for row in all_rows:
        seen[row] = seen.get(row, 0) + 1
    n_exact_dup_rows = sum(c - 1 for c in seen.values() if c > 1)
    n_dup_groups = sum(1 for c in seen.values() if c > 1)

    print(f"\nMissing (NaN/None) coordinate values: {n_missing_vals}")
    print(f"Exact duplicate rows (sample_id, frame_idx, gt_y, gt_x): {n_exact_dup_rows} "
          f"extra copies across {n_dup_groups} distinct duplicated coordinate(s)")
    if n_dup_groups > 0:
        print("  (Duplicates can be biologically real -- e.g. two distinct tracked cells "
              "occupying the identical rounded pixel in the same frame -- this script reports "
              "them, it does not remove them, since deduplication is a labeling-policy decision "
              "outside this script's scope.)")

    if args.out_csv is None or args.report_only:
        print(f"\n--report-only / no --out-csv given: NOTHING WRITTEN. Re-run with --out-csv "
              f"{stage1_dir / 'gt_coordinates.csv'} to write the file after reviewing this report.")
        return

    out_csv_path = Path(args.out_csv)
    if out_csv_path.exists():
        raise SystemExit(f"BLOCKED: {out_csv_path} already exists. Remove it or choose a "
                          f"different --out-csv path -- this script will not overwrite an "
                          f"existing extraction silently.")
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "frame_idx", "gt_y", "gt_x"])
        w.writerows(all_rows)

    print(f"\nSaved: {out_csv_path}  ({total_extracted} rows)")


if __name__ == "__main__":
    main()
