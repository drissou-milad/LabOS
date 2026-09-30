import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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

CASES_CSV = (
    PROJECT_ROOT
    / "results"
    / "exp05_training_dataset"
    / "run02_20samplesrun13_temporal_audit_expanded"
    / "selected_cases.csv"
)

OUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "exp05_training_dataset"
    / "run02_20samplesrun13_temporal_audit_expanded"
    / "original_frame_visual_audit"
)


def load_gt(sample_id):
    """Read GT coordinates from the exact .geff used by Run13."""
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
    """Load the exact raw frame and reproduce the validated MIP."""
    path = DATASET_ROOT / "train" / f"{sample_id}.zarr"

    volume = zarr.open(path, mode="r")["0"]

    frame = np.asarray(volume[frame_idx])
    mip = frame.max(axis=0)

    return frame, mip


def patch_bounds(y, x, patch_size=32):
    half = patch_size // 2

    return (
        y - half,
        y + half,
        x - half,
        x + half,
    )


def audit_case(case):
    sample = case["sample_id"]
    frame_idx = int(case["center_frame"])
    cy = int(case["candidate_y"])
    cx = int(case["candidate_x"])
    probability = float(case["probability"])
    frozen_nearest = float(case["nearest_gt_distance_px"])

    _, mip = load_mip(sample, frame_idx)

    gt_by_frame = load_gt(sample)
    gt_pts = gt_by_frame.get(frame_idx, np.empty((0, 2)))

    y1, y2, x1, x2 = patch_bounds(cy, cx)

    inside = []

    for gy, gx in gt_pts:
        if y1 <= gy < y2 and x1 <= gx < x2:
            inside.append((gy, gx))

    nearest = None
    nearest_dist = None

    if len(gt_pts):
        distances = np.sqrt(
            (gt_pts[:, 0] - cy) ** 2
            + (gt_pts[:, 1] - cx) ** 2
        )

        idx = int(np.argmin(distances))
        nearest = gt_pts[idx]
        nearest_dist = float(distances[idx])

    # Visualization-only robust limits.
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
        label="Run13 candidate",
    )

    rect = plt.Rectangle(
        (x1, y1),
        32,
        32,
        fill=False,
        linewidth=2,
        linestyle="--",
        label="32×32 candidate patch",
    )
    ax1.add_patch(rect)

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
        f"Candidate ({cy}, {cx}) | p={probability:.6f}"
    )

    ax1.set_xlim(0, mip.shape[1])
    ax1.set_ylim(mip.shape[0], 0)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(loc="upper right", fontsize=8)

    # ------------------------------------------------------------
    # Local view
    # ------------------------------------------------------------
    zoom_radius = 48

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

    if (
        nearest is not None
        and zy1 <= nearest[0] < zy2
        and zx1 <= nearest[1] < zx2
    ):
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
        f"Local view | GT inside 32×32: {len(inside)}"
    )

    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Run13 Expanded Original-Frame Visual Audit\n"
        f"label=negative | frozen temporal nearest GT={frozen_nearest:.2f}px | "
        f"recomputed nearest GT={nearest_dist:.2f}px",
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

    return {
        "sample_id": sample,
        "frame_idx": frame_idx,
        "candidate_y": cy,
        "candidate_x": cx,
        "probability": probability,
        "frozen_temporal_nearest_gt_px": frozen_nearest,
        "recomputed_nearest_gt_px": nearest_dist,
        "n_gt_inside_patch": len(inside),
        "output": str(output_path),
    }


def main():
    print("Run13 expanded original-frame visual audit")
    print("=" * 70)
    print(f"Cases : {CASES_CSV}")
    print(f"Output: {OUT_DIR}")
    print()

    if not CASES_CSV.exists():
        raise SystemExit(f"Missing selected cases file: {CASES_CSV}")

    with CASES_CSV.open("r", newline="", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))

    print(f"Selected cases: {len(cases)}")
    print()

    results = []

    for case in cases:
        result = audit_case(case)
        results.append(result)

        print(
            f"{result['sample_id']} "
            f"frame={result['frame_idx']} "
            f"candidate=({result['candidate_y']},{result['candidate_x']}) "
            f"GT_inside_patch={result['n_gt_inside_patch']} "
            f"nearest_GT={result['recomputed_nearest_gt_px']:.3f} "
            f"-> {result['output']}"
        )

    summary_path = OUT_DIR / "expanded_original_frame_summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Saved: {summary_path}")
    print()
    print(
        "READ-ONLY AUDIT: no model, checkpoint, dataset, "
        "benchmark prediction, or previous audit file modified."
    )


if __name__ == "__main__":
    main()
