import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("results/exp05_training_dataset/run02_20samples")
PATCHES_ROOT = ROOT / "patches"


# --------------------------------------------------
# Load split assignments
# --------------------------------------------------

splits = {}

with open(ROOT / "split_manifest.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        splits[row["sample_id"]] = row["split"]


# --------------------------------------------------
# Load manifest
# --------------------------------------------------

rows = []

with open(ROOT / "manifest.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("patch_path"):
            rows.append(row)


# --------------------------------------------------
# Group rows by sample
# --------------------------------------------------

rows_by_sample = {}

for row in rows:
    sid = row["sample_id"]

    if sid not in rows_by_sample:
        rows_by_sample[sid] = []

    rows_by_sample[sid].append(row)


# --------------------------------------------------
# Compute statistics for each sample
# --------------------------------------------------

stats = {}

for sid, split in splits.items():

    sample_rows = rows_by_sample.get(sid, [])

    print(
        f"Processing {sid} "
        f"({split}, {len(sample_rows)} patches)..."
    )

    pos = sum(
        row["label"] == "positive"
        for row in sample_rows
    )

    neg = sum(
        row["label"] == "negative"
        for row in sample_rows
    )

    values_sum = 0.0
    values_sq_sum = 0.0
    values_n = 0

    global_min = float("inf")
    global_max = float("-inf")

    for row in sample_rows:

        # patch_path in manifest is relative to the patches directory
        patch_path = ROOT / row["patch_path"]

        arr = np.load(
            patch_path
        ).astype(np.float64)

        if not np.isfinite(arr).all():
            raise ValueError(
                f"Non-finite values: {patch_path}"
            )

        values_sum += arr.sum()
        values_sq_sum += np.square(arr).sum()
        values_n += arr.size

        global_min = min(
            global_min,
            float(arr.min())
        )

        global_max = max(
            global_max,
            float(arr.max())
        )

    mean = values_sum / values_n

    variance = (
        values_sq_sum / values_n
        - mean ** 2
    )

    std = max(
        variance,
        0.0
    ) ** 0.5

    stats[sid] = {
        "split": split,
        "patches": len(sample_rows),
        "positive": pos,
        "negative": neg,
        "positive_ratio": (
            pos / len(sample_rows)
            if sample_rows
            else 0
        ),
        "mean": mean,
        "std": std,
        "min": global_min,
        "max": global_max,
    }


# --------------------------------------------------
# Print results
# --------------------------------------------------

print()
print("SAMPLE-LEVEL DISTRIBUTION AUDIT")
print("=" * 130)

print(
    f"{'sample_id':18} "
    f"{'split':6} "
    f"{'patches':>8} "
    f"{'pos':>6} "
    f"{'neg':>7} "
    f"{'pos_ratio':>11} "
    f"{'mean':>12} "
    f"{'std':>12} "
    f"{'min':>10} "
    f"{'max':>10}"
)

print("-" * 130)

for sid, s in sorted(
    stats.items(),
    key=lambda item: (
        item[1]["split"],
        item[0]
    )
):

    print(
        f"{sid:18} "
        f"{s['split']:6} "
        f"{s['patches']:8d} "
        f"{s['positive']:6d} "
        f"{s['negative']:7d} "
        f"{s['positive_ratio']:11.6f} "
        f"{s['mean']:12.3f} "
        f"{s['std']:12.3f} "
        f"{s['min']:10.3f} "
        f"{s['max']:10.3f}"
    )


# --------------------------------------------------
# Save JSON
# --------------------------------------------------

output_path = ROOT / "sample_distribution_audit.json"

output_path.write_text(
    json.dumps(
        stats,
        indent=2
    ),
    encoding="utf-8"
)

print()
print("Saved:")
print(output_path)