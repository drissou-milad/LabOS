from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, center_of_mass
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max


DATASET = Path(r".\data\Fluo-N2DH-GOWT1")
SEQUENCE = "01"

SEQUENCE_DIR = DATASET / SEQUENCE
GT_DIR = DATASET / f"{SEQUENCE}_GT" / "TRA"

SIGMA = 6
THRESHOLD = 10
MIN_DISTANCE = 24

MATCH_DISTANCES = [5, 10, 15, 20, 25, 30]


def load_gt(frame):
    gt_file = GT_DIR / f"man_track{frame:03d}.tif"

    gt = tifffile.imread(gt_file)

    ids = np.unique(gt)
    ids = ids[ids > 0]

    points = np.array([
        center_of_mass(gt == cell_id)
        for cell_id in ids
    ])

    return points


def detect(frame):
    image_file = SEQUENCE_DIR / f"t{frame:03d}.tif"

    image = tifffile.imread(image_file).astype(float)

    if image.ndim == 3:
        image = image.max(axis=0)

    filtered = gaussian_filter(image, sigma=SIGMA)

    return peak_local_max(
        filtered,
        threshold_abs=THRESHOLD,
        min_distance=MIN_DISTANCE,
    )


def evaluate(pred, gt):
    if len(pred) == 0 or len(gt) == 0:
        return []

    distance_matrix = cdist(gt, pred)

    rows, cols = linear_sum_assignment(distance_matrix)

    return distance_matrix[rows, cols]


print("=" * 70)
print("LabOS CTC Detection Diagnostic")
print("=" * 70)

all_distances = []

for frame in range(10):

    gt = load_gt(frame)
    pred = detect(frame)

    distances = evaluate(pred, gt)
    all_distances.extend(distances)

    print(
        f"Frame {frame:02d} | "
        f"GT={len(gt):2d} | "
        f"Pred={len(pred):2d} | "
        f"Matches:"
    )

    for threshold in MATCH_DISTANCES:
        matches = sum(d <= threshold for d in distances)

        precision = matches / len(pred) if len(pred) else 0
        recall = matches / len(gt) if len(gt) else 0

        print(
            f"    <= {threshold:2d}px : "
            f"{matches:2d} matches | "
            f"precision={precision:.3f} | "
            f"recall={recall:.3f}"
        )

    if len(distances):
        print(
            f"    distance median={np.median(distances):.2f}px "
            f"mean={np.mean(distances):.2f}px "
            f"max={np.max(distances):.2f}px"
        )

print()
print("=" * 70)
print("Overall")
print("=" * 70)

if all_distances:
    for threshold in MATCH_DISTANCES:
        matches = sum(d <= threshold for d in all_distances)

        print(
            f"<= {threshold:2d}px : "
            f"{matches}/{len(all_distances)} assignments"
        )

    print(
        f"\nMedian assignment distance: "
        f"{np.median(all_distances):.2f}px"
    )