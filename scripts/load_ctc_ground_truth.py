from pathlib import Path

import numpy as np
import tifffile


def load_ctc_ground_truth(gt_tra_dir):
    gt_tra_dir = Path(gt_tra_dir)
    track_txt = gt_tra_dir / "man_track.txt"

    if not track_txt.exists():
        raise FileNotFoundError(
            f"{track_txt} not found - expected a CTC '<seq>_GT/TRA' folder."
        )

    tracks = {}

    for line in track_txt.read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        label, start, end, parent = (int(x) for x in line.split())

        tracks[label] = {
            "start": start,
            "end": end,
            "parent": parent,
        }

    mask_files = sorted(gt_tra_dir.glob("man_track*.tif"))

    if not mask_files:
        raise FileNotFoundError(
            f"No man_track*.tif frames found in {gt_tra_dir}"
        )

    nodes = []
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

            nodes.append({
                "node_id": node_id,
                "t": frame,
                "y": float(ys.mean()),
                "x": float(xs.mean()),
            })

            label_frame_to_node.setdefault(label, {})[frame] = node_id

    edges = []

    for label, info in tracks.items():
        frames_present = sorted(
            label_frame_to_node.get(label, {}).keys()
        )

        for f1, f2 in zip(frames_present, frames_present[1:]):
            edges.append((
                label_frame_to_node[label][f1],
                label_frame_to_node[label][f2],
            ))
    # Add CTC parent-to-daughter lineage edges.
    for label, info in tracks.items():
        frames_present = sorted(
            label_frame_to_node.get(label, {}).keys()
        )

        parent_label = info["parent"]

        if (
            parent_label
            and parent_label in label_frame_to_node
            and frames_present
        ):
            parent_frames = sorted(
                label_frame_to_node[parent_label].keys()
            )

            if parent_frames:
                edges.append((
                    label_frame_to_node[parent_label][parent_frames[-1]],
                    label_frame_to_node[label][frames_present[0]],
                ))

    return nodes, edges


def load_trackmate_ctc(trackmate_res_dir):
    trackmate_res_dir = Path(trackmate_res_dir)
    track_txt = trackmate_res_dir / "res_track.txt"

    if not track_txt.exists():
        raise FileNotFoundError(
            f"{track_txt} not found - expected a TrackMate CTC export."
        )

    tracks = {}

    for line in track_txt.read_text().splitlines():
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 4:
            raise ValueError(
                f"Invalid TrackMate CTC row: {line!r}"
            )

        label, start, end, parent = (int(x) for x in parts)

        tracks[label] = {
            "start": start,
            "end": end,
            "parent": parent,
        }

    mask_files = sorted(trackmate_res_dir.glob("mask*.tif"))

    if not mask_files:
        raise FileNotFoundError(
            f"No mask*.tif files found in {trackmate_res_dir}"
        )

    nodes = []
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

            node_id = f"tm_{label}_{frame}"

            nodes.append({
                "node_id": node_id,
                "t": frame,
                "y": float(ys.mean()),
                "x": float(xs.mean()),
            })

            label_frame_to_node.setdefault(label, {})[frame] = node_id

    edges = []

    for label, info in tracks.items():
        frames_present = sorted(
            label_frame_to_node.get(label, {}).keys()
        )

        for f1, f2 in zip(frames_present, frames_present[1:]):
            edges.append((
                label_frame_to_node[label][f1],
                label_frame_to_node[label][f2],
            ))

        parent_label = info["parent"]

        if (
            parent_label
            and parent_label in label_frame_to_node
            and frames_present
        ):
            parent_frames = sorted(
                label_frame_to_node[parent_label].keys()
            )

            if parent_frames:
                edges.append((
                    label_frame_to_node[parent_label][parent_frames[-1]],
                    label_frame_to_node[label][frames_present[0]],
                ))

    return nodes, edges


def load_ctc_image_sequence(sequence_dir):
    sequence_dir = Path(sequence_dir)

    frame_files = sorted(
        sequence_dir.glob("t*.tif")
    )

    if not frame_files:
        raise FileNotFoundError(
            f"No t*.tif frames found in {sequence_dir}"
        )

    frames = [
        tifffile.imread(str(f))
        for f in frame_files
    ]

    volume = np.stack(frames, axis=0)

    if volume.ndim == 3:
        volume = volume[:, None, :, :]

    return volume
