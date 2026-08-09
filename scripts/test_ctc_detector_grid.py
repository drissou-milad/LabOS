from pathlib import Path

import numpy as np
import tifffile

from scipy.ndimage import gaussian_filter, center_of_mass
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max


DATASET = Path(r".\data\Fluo-N2DH-GOWT1")
SEQUENCE = "01"

IMAGE_DIR = DATASET / SEQUENCE
GT_DIR = DATASET / f"{SEQUENCE}_GT" / "TRA"

SIGMAS = [8, 10, 12, 14, 15]
THRESHOLDS = [4, 6, 8, 10, 12]

MIN_DISTANCE = 24


def load_image(frame):
    image = tifffile.imread(
        IMAGE_DIR / f"t{frame:03d}.tif"
    ).astype(float)

    if image.ndim == 3:
        image = image.max(axis=0)

    return image


def load_gt(frame):
    mask = tifffile.imread(
        GT_DIR / f"man_track{frame:03d}.tif"
    )

    labels = np.unique(mask)
    labels = labels[labels > 0]

    return np.array(
        [center_of_mass(mask == label) for label in labels],
        dtype=float,
    )


def detect(image, sigma, threshold):
    smoothed = gaussian_filter(
        image,
        sigma=sigma,
    )

    return peak_local_max(
        smoothed,
        threshold_abs=threshold,
        min_distance=MIN_DISTANCE,
    )


def match_distances(gt, pred):

    if len(gt) == 0 or len(pred) == 0:
        return np.array([])

    D = np.linalg.norm(
        gt[:, None, :] - pred[None, :, :],
        axis=2,
    )

    rows, cols = linear_sum_assignment(D)

    return D[rows, cols]


images = [load_image(i) for i in range(10)]
gts = [load_gt(i) for i in range(10)]


print()
print("CTC Detector Grid Search")
print("=" * 95)

print(
    f"{'Sigma':>6} "
    f"{'Thresh':>8} "
    f"{'AvgDet':>8} "
    f"{'<=10':>7} "
    f"{'<=15':>7} "
    f"{'<=20':>7} "
    f"{'Median':>9}"
)

print("-" * 95)


for sigma in SIGMAS:

    for threshold in THRESHOLDS:

        detections = []
        all_distances = []

        for image, gt in zip(images, gts):

            pred = detect(
                image,
                sigma,
                threshold,
            )

            detections.append(len(pred))

            d = match_distances(gt, pred)

            if len(d):
                all_distances.extend(d.tolist())

        d = np.asarray(all_distances)

        print(
            f"{sigma:6.1f} "
            f"{threshold:8.1f} "
            f"{np.mean(detections):8.1f} "
            f"{np.sum(d <= 10):7} "
            f"{np.sum(d <= 15):7} "
            f"{np.sum(d <= 20):7} "
            f"{np.median(d):9.2f}"
        )