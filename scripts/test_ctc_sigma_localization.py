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

THRESHOLD = 10
MIN_DISTANCE = 24

SIGMAS = [6, 8, 10, 12, 15]


def load_gt(frame):
    path = GT_DIR / f"man_track{frame:03d}.tif"

    mask = tifffile.imread(path)

    labels = np.unique(mask)
    labels = labels[labels > 0]

    centers = np.array(
        [center_of_mass(mask == label) for label in labels],
        dtype=float,
    )

    return centers


def load_image(frame):
    path = IMAGE_DIR / f"t{frame:03d}.tif"

    image = tifffile.imread(path).astype(float)

    if image.ndim == 3:
        image = image.max(axis=0)

    return image


def detect(image, sigma):
    smoothed = gaussian_filter(image, sigma=sigma)

    return peak_local_max(
        smoothed,
        threshold_abs=THRESHOLD,
        min_distance=MIN_DISTANCE,
    )


def distances(gt, pred):
    if len(gt) == 0 or len(pred) == 0:
        return np.array([])

    D = np.linalg.norm(
        gt[:, None, :] - pred[None, :, :],
        axis=2,
    )

    rows, cols = linear_sum_assignment(D)

    return D[rows, cols]


print()
print("CTC Sigma Localization Test")
print("=" * 75)
print(
    f"{'Sigma':>6} "
    f"{'AvgDet':>8} "
    f"{'<=10px':>8} "
    f"{'<=15px':>8} "
    f"{'<=20px':>8} "
    f"{'<=25px':>8} "
    f"{'Median':>10}"
)
print("-" * 75)


images = [load_image(i) for i in range(10)]
gts = [load_gt(i) for i in range(10)]


for sigma in SIGMAS:

    total_detections = []
    all_distances = []

    for image, gt in zip(images, gts):

        pred = detect(image, sigma)

        total_detections.append(len(pred))

        d = distances(gt, pred)

        if len(d):
            all_distances.extend(d.tolist())

    d = np.asarray(all_distances)

    print(
        f"{sigma:6.1f} "
        f"{np.mean(total_detections):8.1f} "
        f"{np.sum(d <= 10):8} "
        f"{np.sum(d <= 15):8} "
        f"{np.sum(d <= 20):8} "
        f"{np.sum(d <= 25):8} "
        f"{np.median(d):10.2f}"
    )