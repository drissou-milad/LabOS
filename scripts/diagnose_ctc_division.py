from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max

from src import config


DATASET = Path(r".\data\Fluo-N2DH-GOWT1")
SEQ = "01"

GT_DIR = DATASET / "01_GT" / "TRA"
IMG_DIR = DATASET / "01"

SIGMA = config.GAUSSIAN_SIGMA
THRESHOLD = config.DETECTION_THRESHOLD
MIN_DISTANCE = config.CELL_RADIUS


def detect(frame):
    path = IMG_DIR / f"t{frame:03d}.tif"

    image = tifffile.imread(path).astype(float)

    # CTC sequence is (Z, Y, X); this dataset has Z=1.
    if image.ndim == 3:
        image = image.max(axis=0)

    smoothed = gaussian_filter(image, sigma=SIGMA)

    return peak_local_max(
        smoothed,
        threshold_abs=THRESHOLD,
        min_distance=MIN_DISTANCE,
    )


def load_gt_mask(frame):
    path = GT_DIR / f"man_track{frame:03d}.tif"
    return tifffile.imread(path)


def centers_for_track(mask, track_id):
    ys, xs = np.where(mask == track_id)

    if len(ys) == 0:
        return None

    return np.array([ys.mean(), xs.mean()], dtype=float)


print()
print("LabOS CTC Division Diagnostic")
print("=" * 60)
print(f"Sigma       : {SIGMA}")
print(f"Threshold   : {THRESHOLD}")
print(f"Min distance: {MIN_DISTANCE}")
print()

# First GT division:
#
# Track 10: parent, frames 0-46
# Track 12: child, frame 47
# Track 14: child, frames 47-49
#
# Therefore inspect 46 -> 47.

parent_id = 10
child_ids = [12, 14]

for frame in [45, 46, 47, 48, 49]:
    mask = load_gt_mask(frame)
    detections = detect(frame)

    print(f"\nFRAME {frame:02d}")
    print("-" * 60)

    print(f"GT objects represented in division:")
    for track_id in [parent_id] + child_ids:
        center = centers_for_track(mask, track_id)

        if center is None:
            print(f"  Track {track_id}: NOT PRESENT")
        else:
            print(
                f"  Track {track_id}: "
                f"(y={center[0]:.2f}, x={center[1]:.2f})"
            )

    print(f"\nLabOS detections: {len(detections)}")

    if len(detections):
        print("  " + ", ".join(
            f"({float(y):.1f},{float(x):.1f})"
            for y, x in detections
        ))

    # For the division transition, calculate distances from GT
    # parent/daughter centers to every LabOS detection.
    if frame in [46, 47]:
        gt_ids = [parent_id] if frame == 46 else child_ids

        for track_id in gt_ids:
            center = centers_for_track(mask, track_id)

            if center is None or len(detections) == 0:
                continue

            distances = cdist(
                center.reshape(1, 2),
                np.asarray(detections, dtype=float),
            )[0]

            order = np.argsort(distances)[:5]

            print(f"\nNearest LabOS detections to GT Track {track_id}:")
            for rank, idx in enumerate(order, start=1):
                y, x = detections[idx]
                print(
                    f"  #{rank}: distance={distances[idx]:.2f}px "
                    f"at (y={y:.1f}, x={x:.1f})"
                )

print()
print("=" * 60)
print("Division transition: frame 46 -> frame 47")
print()
print("GT:")
print("  Parent : Track 10")
print("  Child  : Track 12")
print("  Child  : Track 14")
print()
print("Interpretation:")
print("  Compare the nearest LabOS detection distances above.")
print("  The current LineageBuilder division radius is:")
print(f"      {config.DIVISION_MAX_DISTANCE:.2f}px")
print()