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
    / "run13_temporal_audit"
)

SAMPLE_ID = "44b6_1d530831"

# These are the original Run13 candidate coordinates.
# They remain fixed across neighboring frames for visual comparison.
CASES = [
    {
        "name": "frame035",
        "center_frame": 35,
        "candidate_y": 56,
        "candidate_x": 211,
        "probability": 0.648115,
        "nearest_gt_distance_px": 167.047897,
    },
    {
        "name": "frame055",
        "center_frame": 55,
        "candidate_y": 168,
        "candidate_x": 73,
        "probability": 0.635201,
        "nearest_gt_distance_px": 14.035669,
    },
]

FRAME_RADIUS = 3
LOCAL_RADIUS = 32  # 64x64 local view


def load_gt(sample_id):
    """Read annotations from the original BioHub GEFF."""
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
        t: np.asarray(points, dtype=float).reshape(-1, 2)
        for t, points in by_frame.items()
    }


def load_volume(sample_id):
    """Open the exact training Zarr volume in read-only mode."""
    path = DATASET_ROOT / "train" / f"{sample_id}.zarr"
    return zarr.open(path, mode="r")["0"]


def get_mip(volume, frame_idx):
    """Reproduce the original MIP operation."""
    frame = np.asarray(volume[frame_idx])
    return frame.max(axis=0)


def get_display_limits(mip):
    """Visualization-only contrast limits."""
    lo = float(np.percentile(mip, 1))
    hi = float(np.percentile(mip, 99.5))

    if hi <= lo:
        hi = lo + 1.0

    return lo, hi


def audit_case(volume, gt_by_frame, case):
    center_frame = case["center_frame"]
    cy = case["candidate_y"]
    cx = case["candidate_x"]

    n_frames = int(volume.shape[0])

    frames = list(
        range(
            max(0, center_frame - FRAME_RADIUS),
            min(n_frames, center_frame + FRAME_RADIUS + 1),
        )
    )

    # Fixed spatial window, centered on the original candidate.
    y1 = max(0, cy - LOCAL_RADIUS)
    y2 = min(256, cy + LOCAL_RADIUS)
    x1 = max(0, cx - LOCAL_RADIUS)
    x2 = min(256, cx + LOCAL_RADIUS)

    fig, axes = plt.subplots(
        2,
        len(frames),
        figsize=(4 * len(frames), 9),
        squeeze=False,
    )

    for col, frame_idx in enumerate(frames):
        mip = get_mip(volume, frame_idx)
        gt_pts = gt_by_frame.get(
            frame_idx,
            np.empty((0, 2), dtype=float),
        )

        lo, hi = get_display_limits(mip)

        # Top row: full MIP.
        ax_full = axes[0, col]
        ax_full.imshow(
            mip,
            cmap="gray",
            vmin=lo,
            vmax=hi,
            origin="upper",
        )

        ax_full.scatter(
            [cx],
            [cy],
            marker="x",
            s=90,
            color="red",
            linewidths=2,
            label="Fixed candidate",
        )

        if len(gt_pts):
            ax_full.scatter(
                gt_pts[:, 1],
                gt_pts[:, 0],
                s=24,
                facecolors="none",
                edgecolors="lime",
                linewidths=1,
                label="GT",
            )

        ax_full.set_title(f"Frame {frame_idx}")
        ax_full.set_xlim(0, 256)
        ax_full.set_ylim(256, 0)
        ax_full.set_xticks([])
        ax_full.set_yticks([])

        if col == 0:
            ax_full.set_ylabel("Full MIP")

        # Bottom row: fixed local crop.
        ax_local = axes[1, col]
        local_mip = mip[y1:y2, x1:x2]

        ax_local.imshow(
            local_mip,
            cmap="gray",
            vmin=lo,
            vmax=hi,
            origin="upper",
            extent=(x1, x2, y2, y1),
        )

        ax_local.scatter(
            [cx],
            [cy],
            marker="x",
            s=100,
            color="red",
            linewidths=2,
            label="Candidate",
        )

        local_gt = gt_pts[
            (gt_pts[:, 0] >= y1)
            & (gt_pts[:, 0] < y2)
            & (gt_pts[:, 1] >= x1)
            & (gt_pts[:, 1] < x2)
        ]

        if len(local_gt):
            ax_local.scatter(
                local_gt[:, 1],
                local_gt[:, 0],
                s=65,
                facecolors="none",
                edgecolors="lime",
                linewidths=1.5,
                label="GT",
            )

        ax_local.set_xlim(x1, x2)
        ax_local.set_ylim(y2, y1)
        ax_local.set_title(
            f"Local view | GT points: {len(local_gt)}"
        )
        ax_local.set_xlabel("x")

        if col == 0:
            ax_local.set_ylabel("Local crop")

    fig.suptitle(
        f"Run13 Temporal Context — {SAMPLE_ID}\n"
        f"Original candidate frame={center_frame}, "
        f"coordinate=({cy},{cx}), "
        f"probability={case['probability']:.6f}\n"
        f"Nearest GT at candidate frame="
        f"{case['nearest_gt_distance_px']:.2f}px\n"
        f"Red cross = fixed candidate coordinate; "
        f"green circles = annotations in each frame",
        fontsize=13,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.88))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUT_DIR / (
        f"{SAMPLE_ID}_{case['name']}_temporal.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(
        f"{case['name']}: frames {frames[0]}-{frames[-1]} "
        f"-> {output_path}"
    )


def main():
    print("Run13 temporal-context audit")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Sample : {SAMPLE_ID}")
    print(f"Output : {OUT_DIR}")
    print("Mode   : read-only; original data and results unchanged")
    print()

    gt_by_frame = load_gt(SAMPLE_ID)
    volume = load_volume(SAMPLE_ID)

    print(f"Volume shape: {volume.shape}")
    print(f"Volume dtype: {volume.dtype}")
    print()

    for case in CASES:
        audit_case(volume, gt_by_frame, case)

    print()
    print("Temporal audit complete.")


if __name__ == "__main__":
    main()
