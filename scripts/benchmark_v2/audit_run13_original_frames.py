import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = Path(
    os.environ.get(
        "BIOHUB_DATASET_PATH",
        r"D:\Datasets\BioHub\biohub-cell-tracking-during-development",
    )
)

OUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "exp05_training_dataset"
    / "run02_20samples"
    / "run13_original_frame_audit"
)

CASES = [
    {
        "sample_id": "44b6_1d530831",
        "frame_idx": 55,
        "candidate_y": 168,
        "candidate_x": 73,
        "probability": 0.635201,
        "label": "negative",
        "nearest_gt_distance_px": 14.035669,
    },
    {
        "sample_id": "44b6_33b596bf",
        "frame_idx": 43,
        "candidate_y": 109,
        "candidate_x": 89,
        "probability": 0.655425,
        "label": "negative",
        "nearest_gt_distance_px": 58.008620,
    },
    {
        "sample_id": "44b6_1d530831",
        "frame_idx": 92,
        "candidate_y": 139,
        "candidate_x": 143,
        "probability": 0.640472,
        "label": "negative",
        "nearest_gt_distance_px": 18.110770,
    },
    {
        "sample_id": "44b6_1d530831",
        "frame_idx": 35,
        "candidate_y": 56,
        "candidate_x": 211,
        "probability": 0.648115,
        "label": "negative",
        "nearest_gt_distance_px": 167.047897,
    },
]


def load_gt(sample_id):
    """Read GT coordinates from the exact .geff used by Run13 dataset construction."""
    path = DATASET_ROOT / "train" / f"{sample_id}.geff"
    graph = zarr.open_group(path, mode="r")

    props = graph["nodes"]["props"]

    ts = np.asarray(props["t"]["values"][:])
    ys = np.asarray(props["y"]["values"][:])
    xs = np.asarray(props["x"]["values"][:])

    by_frame = {}

    for t, y, x in zip(ts, ys, xs):
        by_frame.setdefault(int(t), []).append((float(y), float(x)))

    return {
        frame: np.asarray(points, dtype=float)
        for frame, points in by_frame.items()
    }


def load_mip(sample_id, frame_idx):
    """Load the exact raw volume used by dataset construction and reproduce its MIP."""
    path = DATASET_ROOT / "train" / f"{sample_id}.zarr"

    volume = zarr.open(path, mode="r")["0"]

    frame = np.asarray(volume[frame_idx])
    mip = frame.max(axis=0)

    return frame, mip


def patch_bounds(y, x, patch_size=32, shape=(256, 256)):
    """Return the exact 32x32 extraction bounds used by extract_patch."""
    half = patch_size // 2

    y1 = y - half
    y2 = y + half
    x1 = x - half
    x2 = x + half

    return y1, y2, x1, x2


def audit_case(case):
    sample = case["sample_id"]
    frame_idx = case["frame_idx"]
    cy = case["candidate_y"]
    cx = case["candidate_x"]

    frame, mip = load_mip(sample, frame_idx)

    gt_by_frame = load_gt(sample)
    gt_pts = gt_by_frame.get(frame_idx, np.empty((0, 2)))

    # Find all GT points inside the exact 32x32 candidate patch.
    y1, y2, x1, x2 = patch_bounds(cy, cx)

    inside = []
    for gy, gx in gt_pts:
        if y1 <= gy < y2 and x1 <= gx < x2:
            inside.append((gy, gx))

    # Find nearest GT independently for this visual audit.
    nearest = None
    nearest_dist = None

    if len(gt_pts):
        distances = np.sqrt(
            (gt_pts[:, 0] - cy) ** 2 +
            (gt_pts[:, 1] - cx) ** 2
        )
        idx = int(np.argmin(distances))
        nearest = gt_pts[idx]
        nearest_dist = float(distances[idx])

    # Robust display limits ONLY for visualization.
    # Raw MIP values remain unchanged.
    lo = float(np.percentile(mip, 1))
    hi = float(np.percentile(mip, 99.5))

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    # ------------------------------------------------------------
    # Full original MIP
    # ------------------------------------------------------------
    ax1.imshow(
        mip,
        cmap="gray",
        vmin=lo,
        vmax=hi,
        origin="upper",
    )

    ax1.scatter(
        [cx],
        [cy],
        marker="x",
        s=120,
        linewidths=2,
        label="Candidate",
    )

    # Exact 32x32 patch boundary
    rect = plt.Rectangle(
        (x1, y1),
        32,
        32,
        fill=False,
        linewidth=2,
        linestyle="--",
    )
    ax1.add_patch(rect)

    # All GT points from this frame
    if len(gt_pts):
        ax1.scatter(
            gt_pts[:, 1],
            gt_pts[:, 0],
            s=35,
            facecolors="none",
            edgecolors="lime",
            linewidths=1.2,
            label=f"GT ({len(gt_pts)})",
        )

    # Highlight nearest GT
    if nearest is not None:
        ax1.scatter(
            [nearest[1]],
            [nearest[0]],
            marker="+",
            s=180,
            linewidths=2,
            label=f"Nearest GT ({nearest_dist:.1f}px)",
        )

    ax1.set_title(
        f"{sample} — frame {frame_idx}\n"
        f"Candidate ({cy}, {cx}) | p={case['probability']:.6f}"
    )

    ax1.set_xlim(0, mip.shape[1])
    ax1.set_ylim(mip.shape[0], 0)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(loc="upper right", fontsize=8)

    # ------------------------------------------------------------
    # Local candidate region
    # ------------------------------------------------------------
    zoom_radius = 32

    zy1 = max(0, cy - zoom_radius)
    zy2 = min(mip.shape[0], cy + zoom_radius)
    zx1 = max(0, cx - zoom_radius)
    zx2 = min(mip.shape[1], cx + zoom_radius)

    local = mip[zy1:zy2, zx1:zx2]

    ax2.imshow(
        local,
        cmap="gray",
        vmin=lo,
        vmax=hi,
        origin="upper",
        extent=(zx1, zx2, zy2, zy1),
    )

    ax2.scatter(
        [cx],
        [cy],
        marker="x",
        s=140,
        linewidths=2,
        label="Candidate",
    )

    rect2 = plt.Rectangle(
        (x1, y1),
        32,
        32,
        fill=False,
        linewidth=2,
        linestyle="--",
    )
    ax2.add_patch(rect2)

    # GTs visible in local region
    local_gt = [
        (gy, gx)
        for gy, gx in gt_pts
        if zy1 <= gy < zy2 and zx1 <= gx < zx2
    ]

    if local_gt:
        lgt = np.asarray(local_gt)

        ax2.scatter(
            lgt[:, 1],
            lgt[:, 0],
            s=50,
            facecolors="none",
            edgecolors="lime",
            linewidths=1.5,
            label="GT",
        )

    if nearest is not None and zy1 <= nearest[0] < zy2 and zx1 <= nearest[1] < zx2:
        ax2.scatter(
            [nearest[1]],
            [nearest[0]],
            marker="+",
            s=200,
            linewidths=2,
            label=f"Nearest ({nearest_dist:.1f}px)",
        )

    ax2.set_xlim(zx1, zx2)
    ax2.set_ylim(zy2, zy1)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    ax2.set_title(
        f"Local view\n"
        f"GT inside 32×32: {len(inside)}"
    )

    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Run13 Original-Frame Audit\n"
        f"label={case['label']} | "
        f"nearest GT={case['nearest_gt_distance_px']:.2f}px",
        fontsize=12,
    )

    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    output_name = (
        f"{sample}_frame{frame_idx:03d}"
        f"_y{cy:03d}_x{cx:03d}.png"
    )

    output_path = OUT_DIR / output_name

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"{sample} frame={frame_idx} "
        f"candidate=({cy},{cx}) "
        f"GT_inside_patch={len(inside)} "
        f"nearest_GT={nearest_dist if nearest_dist is not None else 'NONE'} "
        f"-> {output_path}"
    )


def main():
    print("Run13 original-frame visual audit")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Output : {OUT_DIR}")
    print()

    for case in CASES:
        audit_case(case)

    print()
    print("Audit complete.")


if __name__ == "__main__":
    main()
