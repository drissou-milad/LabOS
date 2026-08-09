from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, label
from skimage.measure import regionprops


p = Path(r".\data\Fluo-N2DH-GOWT1\01")

a = tifffile.imread(p / "t047.tif").squeeze().astype(float)

y0, x0 = 809, 745
r = 80

crop = a[y0-r:y0+r+1, x0-r:x0+r+1]

print("REGION-BASED DIVISION TEST")
print("=" * 60)
print("GT child12 = (821, 767)")
print("GT child14 = (795, 801)")
print()

for sigma in [0, 2, 4, 6]:

    if sigma == 0:
        smoothed = crop
    else:
        smoothed = gaussian_filter(crop, sigma=sigma)

    for threshold in [2, 3, 4, 5, 6, 8]:

        binary = smoothed > threshold
        labels, n = label(binary)

        props = regionprops(
            labels,
            intensity_image=smoothed,
        )

        candidates = []

        for region in props:

            if 10 <= region.area <= 200:

                cy, cx = region.centroid

                candidates.append(
                    (
                        round(cy + y0 - r, 1),
                        round(cx + x0 - r, 1),
                        region.area,
                        round(float(region.max_intensity), 1),
                    )
                )

        candidates.sort(
            key=lambda z: z[2],
            reverse=True,
        )

        print(
            f"sigma={sigma}, "
            f"threshold={threshold}, "
            f"regions={n}"
        )

        for candidate in candidates[:20]:
            print("   ", candidate)

        print()