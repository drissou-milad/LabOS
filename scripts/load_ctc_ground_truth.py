"""
Converts a Cell Tracking Challenge (CTC, celltrackingchallenge.net) format ground-truth folder
into the (nodes, edges) shape src/evaluate.py's score_method() expects — the same conversion
LabOS's own src/lineage.py::LineageResult.to_ctc_tracks() does in reverse (LabOS's own results
-> CTC format for submission). Loading real CTC ground truth is the mirror operation: someone
else's CTC-format annotations -> LabOS's internal node/edge graph, so both sides can be scored
with the same src/evaluate.py machinery used everywhere else in this project.

CTC "TRA" (tracking) ground truth, per https://celltrackingchallenge.net/datasets/, is:
  - man_track000.tif, man_track001.tif, ... — one label-mask image per frame. Pixel value 0 is
    background; any other value N means "this pixel belongs to track N in this frame."
  - man_track.txt — one row per track: `label start_frame end_frame parent_label` (whitespace-
    separated ints). parent_label 0 means "no parent" (same convention LineageResult.
    to_ctc_tracks() already uses). A division shows up as two new tracks whose parent_label is
    the track that just ended — the mother's own label is NOT reused for either daughter.

This has NOT been run against a real downloaded CTC dataset in this environment (no network
access here) — tested against tests/fixtures/ctc_sample/, a small hand-authored fixture built
to the schema above, in the same spirit as tests/fixtures/trackmate_sample.xml. Run it against
a real download (see RUNBOOK.md's CTC step) before trusting a comparison table built from it.
"""

from pathlib import Path

import numpy as np
import tifffile


def load_ctc_ground_truth(gt_tra_dir):
    """
    gt_tra_dir: path to a CTC "<seq>_GT/TRA" folder (e.g. ".../Fluo-N2DL-HeLa/01_GT/TRA").

    Returns (nodes, edges) in the same shape src/benchmark.py's load_trackmate_xml() and
    scripts/run_real_benchmark.py's load_ground_truth() both produce, so all three can feed
    the same src/evaluate.py::score_method() / src/benchmark.py::score_external_method().

    node_id here is f"{label}_{frame}" (a string) — deliberately NOT matched against a
    prediction's own node IDs by equality (LabOS's own track IDs and CTC's labels are assigned
    independently and will never coincide). src/evaluate.py::match_nodes() matches spatially
    (nearest neighbor within max_distance), not by ID, so this is safe — see that function's
    docstring before assuming otherwise.
    """
    gt_tra_dir = Path(gt_tra_dir)
    track_txt = gt_tra_dir / "man_track.txt"
    if not track_txt.exists():
        raise FileNotFoundError(
            f"{track_txt} not found — expected a CTC '<seq>_GT/TRA' folder "
            f"(man_track.txt + man_track*.tif). Got: {gt_tra_dir}"
        )

    tracks = {}  # label -> {"start": int, "end": int, "parent": int}
    for line in track_txt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        label, start, end, parent = (int(x) for x in line.split())
        tracks[label] = {"start": start, "end": end, "parent": parent}

    mask_files = sorted(gt_tra_dir.glob("man_track*.tif"))
    if not mask_files:
        raise FileNotFoundError(f"No man_track*.tif frames found in {gt_tra_dir}")

    nodes = []
    # label -> {frame: node_id}, built while scanning frames, used below to wire up edges
    # without a second pass over the mask images.
    label_frame_to_node = {}
    for frame, mask_path in enumerate(mask_files):
        mask = tifffile.imread(str(mask_path))
        for label in np.unique(mask):
            label = int(label)
            if label == 0:
                continue
            ys, xs = np.where(mask == label)
            if ys.size == 0:
                continue
            node_id = f"{label}_{frame}"
            nodes.append({"node_id": node_id, "t": frame, "y": float(ys.mean()), "x": float(xs.mean())})
            label_frame_to_node.setdefault(label, {})[frame] = node_id

    edges = []
    for label, info in tracks.items():
        frames_present = sorted(label_frame_to_node.get(label, {}).keys())
        # Within-track edges: consecutive frames this label actually appears in (usually every
        # frame from start to end, but match reality rather than assuming no gaps).
        for f1, f2 in zip(frames_present, frames_present[1:]):
            edges.append((label_frame_to_node[label][f1], label_frame_to_node[label][f2]))

        # Division edge: this track's mother (info["parent"]) -> this track's first frame.
        parent_label = info["parent"]
        if parent_label and parent_label in label_frame_to_node and frames_present:
            parent_frames = sorted(label_frame_to_node[parent_label].keys())
            if parent_frames:
                edges.append((label_frame_to_node[parent_label][parent_frames[-1]],
                               label_frame_to_node[label][frames_present[0]]))

    return nodes, edges


def load_ctc_image_sequence(sequence_dir):
    """
    sequence_dir: path to a CTC raw-image sequence folder (e.g. ".../Fluo-N2DL-HeLa/01"),
    containing t000.tif, t001.tif, ... 2D frames. Stacked into LabOS's (T, Z=1, Y, X) volume
    shape — CTC's 2D datasets have no Z axis, so it's added the same way
    mvp/pipeline.py::load_volume_from_bytes() adds one for a 3D (T, Y, X) TIFF stack.
    """
    sequence_dir = Path(sequence_dir)
    frame_files = sorted(sequence_dir.glob("t*.tif"))
    if not frame_files:
        raise FileNotFoundError(f"No t*.tif frames found in {sequence_dir}")
    frames = [tifffile.imread(str(f)) for f in frame_files]
    volume = np.stack(frames, axis=0)
    if volume.ndim == 3:  # (T, Y, X) -> (T, Z=1, Y, X)
        volume = volume[:, None, :, :]
    return volume


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gt_tra_dir", help="Path to a CTC '<seq>_GT/TRA' folder")
    args = parser.parse_args()

    nodes, edges = load_ctc_ground_truth(args.gt_tra_dir)
    n_frames = len({n["t"] for n in nodes})
    print(f"{len(nodes)} nodes across {n_frames} frame(s), {len(edges)} edges")
    from collections import Counter
    out_degree = Counter(src for src, _ in edges)
    n_divisions = sum(1 for count in out_degree.values() if count >= 2)
    print(f"{n_divisions} division(s) (nodes with 2+ outgoing edges)")
