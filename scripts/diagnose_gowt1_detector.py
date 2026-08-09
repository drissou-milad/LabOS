import tifffile
import numpy as np
from scipy.ndimage import gaussian_filter, center_of_mass
from skimage.feature import peak_local_max
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# ---------------------------------------------------------
# Load image
# ---------------------------------------------------------
image_path = sorted(
    Path(r".\data\Fluo-N2DH-GOWT1\01").glob("*.tif")
)[0]

x = tifffile.imread(image_path).astype(float)

# ---------------------------------------------------------
# Load ground truth
# ---------------------------------------------------------
gt_path = Path(
    r".\data\Fluo-N2DH-GOWT1\01_GT\TRA\man_track000.tif"
)

gt = tifffile.imread(gt_path)

ids = np.unique(gt)
ids = ids[ids > 0]

gtpts = np.array([
    center_of_mass(gt == i)
    for i in ids
])

print("Image:", image_path)
print("GT:", len(gtpts))
print()
print("Sigma Threshold MinDist Det Matches Precision Recall")
print("-" * 65)

# ---------------------------------------------------------
# Grid search
# ---------------------------------------------------------
for sigma in [2, 3, 4, 5, 6]:

    y = gaussian_filter(x, sigma)

    for threshold in [10, 15, 20, 25, 30]:

        for min_distance in [16, 20, 24, 28]:

            pred = peak_local_max(
                y,
                threshold_abs=threshold,
                min_distance=min_distance,
            )

            n_pred = len(pred)

            if n_pred == 0:
                print(
                    f"{sigma:5} {threshold:9} {min_distance:8} "
                    f"{0:3} {0:7} {0.000:9} {0.000:7}"
                )
                continue

            # Distance matrix: GT -> predictions
            D = cdist(gtpts, pred)

            # Hungarian matching
            rows, cols = linear_sum_assignment(D)

            matched = np.sum(D[rows, cols] <= 15)

            precision = matched / n_pred
            recall = matched / len(gtpts)

            print(
                f"{sigma:5} {threshold:9} {min_distance:8} "
                f"{n_pred:3} {matched:7} "
                f"{precision:9.3f} {recall:7.3f}"
            )
