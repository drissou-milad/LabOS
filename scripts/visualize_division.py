from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max


# ============================================================
# LABOS — Division Visualization
# ============================================================

DATA_DIR = Path(r".\data\Fluo-N2DH-GOWT1\01")
OUTPUT_DIR = Path(r".\outputs\division_diagnostic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Suspected division
PARENT = (46, 809.45, 744.90)

CHILDREN = [
    (47, 821.00, 767.00),
    (47, 795.00, 801.00),
]

FRAMES = [44, 45, 46, 47, 48, 49]

SIGMAS = [1, 2, 3]

THRESHOLD = 3
MIN_DISTANCE = 8


def load_frame(frame):
    path = DATA_DIR / f"t{frame:03d}.tif"

    image = tifffile.imread(path).squeeze()

    return image.astype(float)


def detect(image, sigma):
    filtered = gaussian_filter(image, sigma=sigma)

    peaks = peak_local_max(
        filtered,
        threshold_abs=THRESHOLD,
        min_distance=MIN_DISTANCE,
        num_peaks=100,
    )

    return filtered, peaks


def draw_target(ax, y, x, label, marker="x"):
    ax.scatter(
        x,
        y,
        marker=marker,
        s=100,
        linewidths=2,
        label=label,
    )


print("=" * 80)
print("LABOS — DIVISION VISUALIZATION")
print("=" * 80)
print()

for frame in FRAMES:

    image = load_frame(frame)

    print(f"Processing frame {frame:03d}")

    fig, axes = plt.subplots(
        1,
        len(SIGMAS) + 1,
        figsize=(18, 5),
    )

    # --------------------------------------------------------
    # Crop around suspected division
    # --------------------------------------------------------

    y_min, y_max = 740, 850
    x_min, x_max = 700, 850

    crop = image[y_min:y_max, x_min:x_max]

    # --------------------------------------------------------
    # Raw image
    # --------------------------------------------------------

    ax = axes[0]

    ax.imshow(
        crop,
        cmap="gray",
        origin="upper",
    )

    ax.set_title(f"Frame {frame} — RAW")

    # Parent
    if frame == PARENT[0]:

        y, x = PARENT[1:]

        draw_target(
            ax,
            y - y_min,
            x - x_min,
            "PARENT",
        )

    # Children
    for child_frame, y, x in CHILDREN:

        if frame == child_frame:

            draw_target(
                ax,
                y - y_min,
                x - x_min,
                "CHILD",
                marker="+",
            )

    ax.legend()

    # --------------------------------------------------------
    # Gaussian + detected peaks
    # --------------------------------------------------------

    for index, sigma in enumerate(SIGMAS, start=1):

        filtered, peaks = detect(
            image,
            sigma,
        )

        ax = axes[index]

        filtered_crop = filtered[
            y_min:y_max,
            x_min:x_max,
        ]

        ax.imshow(
            filtered_crop,
            cmap="gray",
            origin="upper",
        )

        # Plot detected peaks
        for py, px in peaks:

            if (
                y_min <= py < y_max
                and x_min <= px < x_max
            ):

                ax.scatter(
                    px - x_min,
                    py - y_min,
                    s=25,
                    marker="o",
                )

        # Parent
        if frame == PARENT[0]:

            y, x = PARENT[1:]

            draw_target(
                ax,
                y - y_min,
                x - x_min,
                "PARENT",
            )

        # Children
        for child_frame, y, x in CHILDREN:

            if frame == child_frame:

                draw_target(
                    ax,
                    y - y_min,
                    x - x_min,
                    "CHILD",
                    marker="+",
                )

        ax.set_title(
            f"Frame {frame} — σ={sigma}"
        )

    for ax in axes:

        ax.set_xlim(0, x_max - x_min)
        ax.set_ylim(y_max - y_min, 0)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"division_frame_{frame:03d}.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(f"  saved → {output_path}")


print()
print("=" * 80)
print("DONE")
print("=" * 80)
print()
print(f"Images saved in:")
print(OUTPUT_DIR)