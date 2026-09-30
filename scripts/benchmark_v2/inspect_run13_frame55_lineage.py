import os
from pathlib import Path

import numpy as np
import zarr

root = Path(os.environ.get(
    "BIOHUB_DATASET_PATH",
    r"D:\Datasets\BioHub\biohub-cell-tracking-during-development"
))

sample = "44b6_1d530831"
path = root / "train" / f"{sample}.geff"

graph = zarr.open_group(path, mode="r")

print("GEFF groups and arrays:")
def inspect_group(group, prefix=""):
    for key in group.keys():
        obj = group[key]
        name = f"{prefix}/{key}"
        if isinstance(obj, zarr.Group):
            print("GROUP", name)
            inspect_group(obj, name)
        else:
            print("ARRAY", name, "shape=", obj.shape, "dtype=", obj.dtype)

inspect_group(graph)

print("\nRoot attributes:")
print(dict(graph.attrs))

print("\nNode properties:")
props = graph["nodes"]["props"]
print(list(props.keys()))

ts = np.asarray(props["t"]["values"][:])
ys = np.asarray(props["y"]["values"][:])
xs = np.asarray(props["x"]["values"][:])

for t in range(52, 59):
    mask = ts == t
    pts = np.column_stack((ys[mask], xs[mask]))

    cy, cx = 168, 73

    print(f"\nFrame {t}:")
    if len(pts) == 0:
        print("No GT points")
        continue

    d = np.sqrt((pts[:, 0] - cy)**2 + (pts[:, 1] - cx)**2)

    for idx in np.argsort(d):
        print(
            f"  y={pts[idx,0]:.3f}, "
            f"x={pts[idx,1]:.3f}, "
            f"distance={d[idx]:.3f}"
        )
