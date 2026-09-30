import os
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(os.environ.get(
    "BIOHUB_DATASET_PATH",
    r"D:\Datasets\BioHub\biohub-cell-tracking-during-development"
))

SAMPLE = "44b6_1d530831"
GEFF = ROOT / "train" / f"{SAMPLE}.geff"

CANDIDATE_Y = 168
CANDIDATE_X = 73

graph = zarr.open_group(GEFF, mode="r")

node_ids = np.asarray(graph["nodes"]["ids"][:])
props = graph["nodes"]["props"]

ts = np.asarray(props["t"]["values"][:])
ys = np.asarray(props["y"]["values"][:])
xs = np.asarray(props["x"]["values"][:])
zs = np.asarray(props["z"]["values"][:])

edges = np.asarray(graph["edges"]["ids"][:])

# Map each GEFF node ID to its position in the node-property arrays.
id_to_idx = {
    int(node_id): i
    for i, node_id in enumerate(node_ids)
}

# Build directed predecessor/successor lookup tables.
parents = {}
children = {}

for source, target in edges:
    source = int(source)
    target = int(target)

    children.setdefault(source, []).append(target)
    parents.setdefault(target, []).append(source)


def describe_node(node_id, indent=""):
    idx = id_to_idx[node_id]

    print(
        f"{indent}node={node_id} "
        f"t={int(ts[idx])} "
        f"(y={int(ys[idx])}, x={int(xs[idx])}, z={int(zs[idx])})"
    )


def print_neighbors(node_id, direction, neighbor_map):
    neighbors = neighbor_map.get(node_id, [])

    print(f"  {direction}: {len(neighbors)}")

    if not neighbors:
        print("    none")
        return

    for neighbor_id in neighbors:
        if neighbor_id in id_to_idx:
            describe_node(neighbor_id, indent="    ")
        else:
            print(f"    node={neighbor_id} (not found in node IDs)")


print("GEFF lineage audit")
print(f"Sample: {SAMPLE}")
print(f"Nodes: {len(node_ids)}")
print(f"Edges: {len(edges)}")
print()

# Find nearest annotated node to the fixed candidate in frames 52-58.
selected = []

for frame in range(52, 59):
    indices = np.where(ts == frame)[0]

    print(f"FRAME {frame}")

    if len(indices) == 0:
        print("  No annotated nodes")
        continue

    distances = np.sqrt(
        (ys[indices] - CANDIDATE_Y) ** 2
        + (xs[indices] - CANDIDATE_X) ** 2
    )

    order = np.argsort(distances)

    for rank, local_idx in enumerate(order[:3], start=1):
        idx = int(indices[local_idx])
        node_id = int(node_ids[idx])

        print(
            f"  rank={rank} "
            f"node={node_id} "
            f"y={int(ys[idx])} "
            f"x={int(xs[idx])} "
            f"z={int(zs[idx])} "
            f"distance={distances[local_idx]:.3f}"
        )

        if rank == 1:
            selected.append(node_id)

    print()

print("=" * 70)
print("DIRECTED NEIGHBORS OF THE NEAREST NODES")
print("=" * 70)

for node_id in selected:
    print()
    describe_node(node_id)
    print_neighbors(node_id, "PARENTS", parents)
    print_neighbors(node_id, "CHILDREN", children)

print()
print("Lineage audit complete. No data modified.")
