"""
Read-only inspector for a sample's .geff ground-truth graph structure.

Does NOT modify src/dataset.py, src/config.py, src/detector.py, matching_v2.py,
run_benchmark_v2.py, or any Benchmark v1 file. Uses BioHubDataset.load_graph() exactly as it
exists today (unmodified) and just prints what's actually inside the returned zarr group,
instead of assuming a schema.

Run this BEFORE trusting any proposed fix to load_ground_truth_points() -- geff's own spec
(https://liveimagetrackingtools.org/geff/v1.1.4.1.1/specification/) nests node properties
under nodes/props/<name>/values rather than flat nodes/<name>, which is the likely cause of the
`KeyError: 't'`, but the exact property NAMES (is time really called "t"? "time"? something
else?) are only authoritative from this specific file's own `.zattrs` axes metadata -- don't
hardcode a guess without checking.

Usage:
    export BIOHUB_DATASET_PATH=/path/to/biohub-cell-tracking-during-development
    python scripts/benchmark_v2/inspect_geff_schema.py --sample 44b6_0113de3b
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import zarr

from src import config
from src.dataset import BioHubDataset


def walk(group, prefix="", max_depth=5, depth=0):
    if depth > max_depth:
        print(f"{prefix}... (max depth reached)")
        return
    try:
        keys = list(group.keys())
    except Exception as e:
        print(f"{prefix}<could not list keys: {e}>")
        return

    for key in keys:
        item = group[key]
        if hasattr(item, "keys"):  # it's a (sub)group
            print(f"{prefix}{key}/")
            walk(item, prefix=prefix + "  ", max_depth=max_depth, depth=depth + 1)
        else:  # it's an array
            try:
                print(f"{prefix}{key}  -- shape={item.shape} dtype={item.dtype}")
            except Exception as e:
                print(f"{prefix}{key}  -- <could not read shape/dtype: {e}>")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True, help="e.g. 44b6_0113de3b")
    args = parser.parse_args()

    dataset = BioHubDataset(config.DATASET_PATH)
    graph = dataset.load_graph(args.sample)

    print("=" * 70)
    print(f"Sample: {args.sample}")
    print("=" * 70)

    print("\n--- .zattrs (top-level metadata; look for a 'geff' key with 'axes') ---")
    try:
        attrs = dict(graph.attrs)
        import json
        print(json.dumps(attrs, indent=2, default=str))
    except Exception as e:
        print(f"<could not read attrs: {e}>")

    print("\n--- full group/array tree ---")
    walk(graph)

    print("\n--- specifically checking the paths run_real_benchmark.py / run_benchmark_v2.py "
          "currently assume ---")
    for path in [("nodes", "ids"), ("nodes", "t"), ("nodes", "y"), ("nodes", "x"),
                 ("nodes", "props"), ("edges", "ids")]:
        try:
            node = graph
            for p in path:
                node = node[p]
            extra = f" shape={node.shape} dtype={node.dtype}" if hasattr(node, "shape") else " (group)"
            print(f"  graph[{path!r}] EXISTS.{extra}")
        except KeyError:
            print(f"  graph[{path!r}] -> KeyError (does not exist)")

    if "nodes" in graph and "props" in graph["nodes"]:
        print("\n--- nodes/props/ subgroup names (these are your real property identifiers) ---")
        for name in graph["nodes"]["props"].keys():
            sub = graph["nodes"]["props"][name]
            sub_keys = list(sub.keys()) if hasattr(sub, "keys") else []
            print(f"  nodes/props/{name}/  -> contains: {sub_keys}")


if __name__ == "__main__":
    main()
