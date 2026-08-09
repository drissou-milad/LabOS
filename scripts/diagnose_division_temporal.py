import tifffile
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max


DATA = Path(r".\data\Fluo-N2DH-GOWT1\01")

# Known GT locations from our benchmark investigation
TARGETS = [
    ("PARENT", 46, 809.45, 744.90),
    ("CHILD12", 47, 821.00, 767.00),
    ("CHILD14", 47, 795.00, 801.00),
]

SIGMAS = [1, 2, 3, 4, 5, 6 , 15]
THRESHOLD = 3
WINDOW = 45


def distance(y1, x1, y2, x2):
    return float(np.hypot(y1 - y2, x1 - x2))


print("=" * 80)
print("LABOS — DIVISION TEMPORAL ANALYSIS")
print("=" * 80)
print()

for name, frame, gy, gx in TARGETS:

    print(
        f"{name}: frame={frame}, "
        f"GT=({gy:.2f}, {gx:.2f})"
    )

    for f in range(max(0, frame - 2), frame + 3):

        image = tifffile.imread(
            DATA / f"t{f:03d}.tif"
        ).squeeze().astype(float)

        y0 = max(0, int(round(gy)) - WINDOW)
        y1 = min(image.shape[0], int(round(gy)) + WINDOW + 1)

        x0 = max(0, int(round(gx)) - WINDOW)
        x1 = min(image.shape[1], int(round(gx)) + WINDOW + 1)

        crop = image[y0:y1, x0:x1]

        print()
        print(f"  FRAME {f:03d}")

        print(
            f"    raw: "
            f"max={crop.max():.1f}, "
            f"mean={crop.mean():.3f}, "
            f"p99={np.percentile(crop, 99):.1f}, "
            f"p99.9={np.percentile(crop, 99.9):.1f}"
        )

        for sigma in SIGMAS:

            smoothed = gaussian_filter(crop, sigma)

            peaks = peak_local_max(
                smoothed,
                threshold_abs=THRESHOLD,
                min_distance=8,
                num_peaks=15,
            )

            candidates = []

            for py, px in peaks:

                ay = py + y0
                ax = px + x0
                value = float(smoothed[py, px])

                d = distance(
                    ay,
                    ax,
                    gy,
                    gx,
                )

                if d <= WINDOW:

                    candidates.append(
                        (
                            d,
                            ay,
                            ax,
                            value,
                        )
                    )

            candidates.sort()

            print(f"    sigma={sigma}")

            if not candidates:
                print("      no candidates")
                continue

            for d, ay, ax, value in candidates[:8]:

                print(
                    f"      "
                    f"({ay:7.2f}, {ax:7.2f}) "
                    f"dist={d:6.2f} "
                    f"value={value:6.3f}"
                )

    print()
    print("-" * 80)