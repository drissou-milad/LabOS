"""
Prints the actual structure of one training sample's ground-truth graph
(BioHubDataset.load_graph()) — run this FIRST, before scripts/run_real_benchmark.py, and read
its output.

Why this script exists: earlier in this project, src/evaluate.py and the notebooks' exploration
of `graph["nodes"]` / `graph["edges"]` assumed a particular field layout (ids/t/z/y/x for nodes;
some pairing for edges) based on what an earlier notebook session printed. That's secondhand
memory of a partial exploration, not a verified schema — good enough to build a converter
against, not good enough to trust blindly. Run this and actually look at what it prints before
trusting scripts/run_real_benchmark.py's conversion logic.

Usage:
    python scripts/inspect_ground_truth.py [--sample SAMPLE_NAME]
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.dataset import BioHubDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default=None, help="Sample name; defaults to the first training sample")
    args = parser.parse_args()

    dataset = BioHubDataset(config.DATASET_PATH)
    sample = args.sample or dataset.train_samples[0]
    print(f"Sample: {sample}\n")

    graph = dataset.load_graph(sample)
    print("Top-level graph keys:", list(graph.keys()) if hasattr(graph, "keys") else type(graph))
    print()

    for group_name in ("nodes", "edges"):
        if group_name not in graph:
            print(f"No '{group_name}' group found.")
            continue
        group = graph[group_name]
        print(f"=== graph['{group_name}'] ===")
        keys = list(group.keys()) if hasattr(group, "keys") else []
        print("  keys:", keys)
        for key in keys:
            arr = group[key][:]
            print(f"  {key}: shape={arr.shape} dtype={arr.dtype} "
                  f"first values={arr[:5].tolist() if len(arr) else '[]'}")
        print()

    print("--- What to do with this output ---")
    print("Compare the 'nodes' keys against what scripts/run_real_benchmark.py assumes")
    print("(node_id/t/z/y/x-style fields) and the 'edges' keys against what it assumes")
    print("(a pair of node-id arrays, or a single Nx2 'ids' array). If they don't match,")
    print("edit the conversion function at the top of run_real_benchmark.py accordingly —")
    print("the actual bipartite-matching and scoring logic underneath it doesn't need to change.")


if __name__ == "__main__":
    main()
