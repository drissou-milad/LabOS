from pathlib import Path

import tifffile
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, center_of_mass
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max


DATASET = Path(r".\data\Fluo-N2DH-GOWT1")
SEQUENCE = "01"

SEQUENCE_DIR = DATASET / SEQUENCE
GT_DIR = DATASET / f"{SEQUENCE}_GT" / "TRA"

SIGMA = 6
THRESHOLD = 10
MIN_DISTANCE = 24

FRAMES = [0, 1, 2, 6, 9]


def load_gt(frame):
    gt_file = GT_DIR / f"man_track{frame:03d}.tif"

    gt = tifffile.imread(gt_file)

    ids = np.unique(gt)
    ids = ids[ids > 0]

    centers = np.array(
        [center_of_mass(gt == i) for i in ids],
        dtype=float,
    )

    return centers


def detect(frame):
    image_file = SEQUENCE_DIR / f"t{frame:03d}.tif"

    image = tifffile.imread(image_file).astype(float)

    # CTC volume may be (Z,Y,X)
    if image.ndim == 3:
        image = image.max(axis=0)

    smoothed = gaussian_filter(image, sigma=SIGMA)

    centers = peak_local_max(
        smoothed,
        threshold_abs=THRESHOLD,
        min_distance=MIN_DISTANCE,
    )

    return image, centers


def match_points(gt, pred, max_distance=150):
    if len(gt) == 0 or len(pred) == 0:
        return []

    D = np.linalg.norm(
        gt[:, None, :] - pred[None, :, :],
        axis=2,
    )

    rows, cols = linear_sum_assignment(D)

    matches = []

    for r, c in zip(rows, cols):
        matches.append(
            (r, c, D[r, c])
        )

    return matches


for frame in FRAMES:

    image, pred = detect(frame)
    gt = load_gt(frame)

    matches = match_points(gt, pred)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(image, cmap="gray")

    # GT = red
    ax.scatter(
        gt[:, 1],
        gt[:, 0],
        s=80,
        facecolors="none",
        edgecolors="red",
        linewidths=1.5,
        label=f"GT ({len(gt)})",
    )

    # Prediction = blue
    ax.scatter(
        pred[:, 1],
        pred[:, 0],
        s=40,
        c="blue",
        marker="x",
        label=f"LabOS ({len(pred)})",
    )

    # Draw correspondence lines
    for r, c, distance in matches:

        if distance <= 30:

            ax.plot(
                [gt[r, 1], pred[c, 1]],
                [gt[r, 0], pred[c, 0]],
                linewidth=0.8,
                alpha=0.6,
            )

    ax.set_title(
        f"Frame {frame:02d} | "
        f"GT={len(gt)} | "
        f"Pred={len(pred)}"
    )

    ax.legend()

    ax.set_axis_off()

    output = Path(f"ctc_overlay_{frame:03d}.png")

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved {output}")