import csv
import os
from pathlib import Path

import numpy as np
import zarr

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = Path(
    os.environ.get(
        "BIOHUB_DATASET_PATH",
        r"D:\Datasets\BioHub\biohub-cell-tracking-during-development",
    )
)

SAMPLE_ID = "44b6_1d530831"

OUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "exp05_training_dataset"
    / "run02_20samples"
    / "run13_temporal_audit"
)

CASES = [
    {
        "case": "frame035",
        "center_frame": 35,
        "candidate_y": 56,
        "candidate_x": 211,
        "probability": 0.648115,
    },
    {
        "case": "frame055",
        "center_frame": 55,
        "candidate_y": 168,
        "candidate_x": 73,
        "probability": 0.635201,
    },
]

FRAME_RADIUS = 3
PATCH_SIZE = 32


def load_gt_by_frame():
    path = DATASET_ROOT / "train" / f"{SAMPLE_ID}.geff"
    graph = zarr.open_group(path, mode="r")
    props = graph["nodes"]["props"]

    ts = np.asarray(props["t"]["values"][:])
    ys = np.asarray(props["y"]["values"][:])
    xs = np.asarray(props["x"]["values"][:])

    by_frame = {}

    for t, y, x in zip(ts, ys, xs):
        by_frame.setdefault(int(t), []).append((float(y), float(x)))

    return {
        t: np.asarray(points, dtype=float).reshape(-1, 2)
        for t, points in by_frame.items()
    }


def main():
    print("Run13 temporal GT-context audit")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Sample : {SAMPLE_ID}")
    print("Mode   : read-only")
    print()

    gt_by_frame = load_gt_by_frame()
    rows = []

    half = PATCH_SIZE // 2

    for case in CASES:
        cy = case["candidate_y"]
        cx = case["candidate_x"]
        center = case["center_frame"]

        y1, y2 = cy - half, cy + half
        x1, x2 = cx - half, cx + half

        for t in range(center - FRAME_RADIUS, center + FRAME_RADIUS + 1):
            pts = gt_by_frame.get(
                t,
                np.empty((0, 2), dtype=float),
            )

            n_gt = len(pts)

            if n_gt:
                distances = np.sqrt(
                    (pts[:, 0] - cy) ** 2
                    + (pts[:, 1] - cx) ** 2
                )

                nearest_idx = int(np.argmin(distances))
                nearest_dist = float(distances[nearest_idx])
                nearest_y = float(pts[nearest_idx, 0])
                nearest_x = float(pts[nearest_idx, 1])

                inside_mask = (
                    (pts[:, 0] >= y1)
                    & (pts[:, 0] < y2)
                    & (pts[:, 1] >= x1)
                    & (pts[:, 1] < x2)
                )

                n_inside = int(inside_mask.sum())

                if n_inside:
                    inside_pts = pts[inside_mask]
                    inside_dists = np.sqrt(
                        (inside_pts[:, 0] - cy) ** 2
                        + (inside_pts[:, 1] - cx) ** 2
                    )
                    nearest_inside_dist = float(inside_dists.min())
                else:
                    nearest_inside_dist = ""
            else:
                nearest_dist = ""
                nearest_y = ""
                nearest_x = ""
                n_inside = 0
                nearest_inside_dist = ""

            rows.append({
                "sample_id": SAMPLE_ID,
                "case": case["case"],
                "center_frame": center,
                "frame_idx": t,
                "candidate_y_fixed": cy,
                "candidate_x_fixed": cx,
                "probability": case["probability"],
                "n_gt_in_frame": n_gt,
                "nearest_gt_distance_to_fixed_candidate_px": nearest_dist,
                "nearest_gt_y": nearest_y,
                "nearest_gt_x": nearest_x,
                "n_gt_inside_fixed_32px_patch": n_inside,
                "nearest_gt_inside_patch_distance_px": nearest_inside_dist,
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "run13_temporal_gt_context.csv"

    fieldnames = list(rows[0].keys())

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Rows written: {len(rows)}")
    print(f"Output: {out_path}")
    print()
    print("Frame-by-frame summary:")

    for row in rows:
        print(
            f"{row['case']} "
            f"frame={row['frame_idx']} "
            f"GT_count={row['n_gt_in_frame']} "
            f"nearest={row['nearest_gt_distance_to_fixed_candidate_px']} "
            f"GT_inside_patch={row['n_gt_inside_fixed_32px_patch']}"
        )

    print()
    print("Temporal GT-context audit complete.")


if __name__ == "__main__":
    main()
