"""
Lineage construction: turns per-frame detections into a full cell-tracking
graph with division events and persistent track IDs.

This does NOT modify HungarianTracker or CellDetector — it composes them.
HungarianTracker already solves the easy 95% of tracking (one cell now is the
same cell next frame); this module adds the harder 5%: when a cell divides,
one detection at frame t is replaced by two detections at frame t+1, and a
plain 1-to-1 assignment can only ever explain one of those two children,
leaving the other one looking like an unrelated new cell.

Algorithm, per consecutive frame pair (t, t+1):

1. Run HungarianTracker.match() as usual — this is the normal 1-to-1 linking
   and handles every cell that didn't just divide.
2. Whatever in frame t+1 that step didn't match ("orphans") gets a second
   pass: each orphan is matched to the nearest frame-t detection within
   DIVISION_MAX_DISTANCE that still has spare capacity to be a parent (a
   parent can have at most 2 children total, across both passes).
3. Any frame-t detection that ends up with exactly 2 children (whether
   1 normal + 1 orphan, or 2 orphans) is a division: both children start
   brand-new track IDs, linked back to the parent's track via
   parent_track_id. A parent with exactly 1 child is a normal continuation
   (same track ID carries over). A parent with 0 children is a track end
   (cell left the frame, died, or a detection was missed).

This mirrors the standard Cell Tracking Challenge (CTC) convention: a
division ends the parent's track and starts two new ones, rather than
letting one daughter silently "inherit" the parent's ID.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from src import config


class LineageNode:
    """A single detection, placed in its track/lineage context."""

    __slots__ = ("node_id", "t", "y", "x", "track_id", "parent_node_id")

    def __init__(self, node_id, t, y, x, track_id, parent_node_id):
        self.node_id = node_id
        self.t = t
        self.y = y
        self.x = x
        self.track_id = track_id
        self.parent_node_id = parent_node_id

    def as_dict(self):
        return {
            "node_id": self.node_id,
            "t": self.t,
            "y": self.y,
            "x": self.x,
            "track_id": self.track_id,
            "parent_node_id": self.parent_node_id,
        }


class LineageResult:
    """
    Output of LineageBuilder.build(): nodes, edges, and division events for
    one sample's full sequence of frames.
    """

    def __init__(self, nodes, edges, divisions, tracks):
        self.nodes = nodes          # list[LineageNode]
        self.edges = edges          # list[(source_node_id, target_node_id)]
        self.divisions = divisions  # list[dict]: parent_node_id, parent_track_id, children_node_ids, children_track_ids, t
        self.tracks = tracks        # dict[track_id] -> {"start_t", "end_t", "parent_track_id"}

    def to_dataframe(self):
        """One row per detection: node_id, t, y, x, track_id, parent_node_id,
        parent_track_id, is_division_parent."""
        import pandas as pd

        columns = ["node_id", "t", "y", "x", "track_id", "parent_node_id",
                   "parent_track_id", "is_division_parent"]
        dividing_parent_ids = {d["parent_node_id"] for d in self.divisions}
        rows = []
        for n in self.nodes:
            rows.append({
                **n.as_dict(),
                "parent_track_id": self.tracks[n.track_id]["parent_track_id"],
                "is_division_parent": n.node_id in dividing_parent_ids,
            })
        # pd.DataFrame([]) has zero columns, not just zero rows — writing that to CSV produces
        # a genuinely empty file that pandas.read_csv can't even parse back (EmptyDataError),
        # breaking mvp/experiments.py's load_lineage_result() the moment an experiment detects
        # nothing at all (a real, reachable case: an overly strict Configuration, or genuinely
        # empty footage). Explicit `columns=` keeps the header row even with zero detections.
        return pd.DataFrame(rows, columns=columns)

    def to_ctc_tracks(self):
        """
        Standard Cell Tracking Challenge track table: one row per track ID
        with (track_id, start_t, end_t, parent_track_id). parent_track_id is
        0 for tracks with no parent (CTC convention: 0 means "no parent").
        """
        import pandas as pd

        rows = []
        for track_id, info in sorted(self.tracks.items()):
            rows.append({
                "track_id": track_id,
                "start_t": info["start_t"],
                "end_t": info["end_t"],
                "parent_track_id": info["parent_track_id"] or 0,
            })
        return pd.DataFrame(rows, columns=["track_id", "start_t", "end_t", "parent_track_id"])

    def n_divisions(self):
        return len(self.divisions)

    @classmethod
    def from_dataframe(cls, df):
        """
        Reconstructs a LineageResult from to_dataframe()'s own output (or an equivalent CSV
        read back from disk) — node_id, t, y, x, track_id, parent_node_id, parent_track_id,
        is_division_parent. This is what makes a persisted experiment's interactive viewer and
        lineage-tree highlighting still work after a server restart: mvp/experiments.py loads
        tracks.csv and calls this instead of needing the original in-memory LineageBuilder
        output.

        Edges and divisions are both re-derived from parent_node_id / is_division_parent —
        to_dataframe() doesn't lose any information from a round-trip perspective.
        """
        import pandas as pd

        def _int_or_none(value):
            return None if pd.isna(value) else int(value)

        nodes = []
        for _, row in df.iterrows():
            nodes.append(LineageNode(
                node_id=int(row["node_id"]),
                t=int(row["t"]),
                y=float(row["y"]),
                x=float(row["x"]),
                track_id=int(row["track_id"]),
                parent_node_id=_int_or_none(row["parent_node_id"]),
            ))

        edges = [
            (n.parent_node_id, n.node_id) for n in nodes if n.parent_node_id is not None
        ]

        tracks = {}
        for track_id, group in df.groupby("track_id"):
            tracks[int(track_id)] = {
                "start_t": int(group["t"].min()),
                "end_t": int(group["t"].max()),
                "parent_track_id": _int_or_none(group["parent_track_id"].iloc[0]),
            }

        divisions = []
        dividing_parents = df[df["is_division_parent"]]
        node_by_id = {n.node_id: n for n in nodes}
        for _, parent_row in dividing_parents.iterrows():
            parent_node_id = int(parent_row["node_id"])
            children = [n for n in nodes if n.parent_node_id == parent_node_id]
            if not children:
                continue  # shouldn't happen for a well-formed export, but don't crash on one
            divisions.append({
                "parent_node_id": parent_node_id,
                "parent_track_id": int(parent_row["track_id"]),
                "children_node_ids": [c.node_id for c in children],
                "children_track_ids": [c.track_id for c in children],
                "t": children[0].t,
            })

        return cls(nodes, edges, divisions, tracks)

    def get_descendant_track_ids(self, track_id, include_self=True):
        """
        All track IDs descended from track_id via divisions (children, grandchildren, ...).
        BFS through self.divisions rather than self.edges, since divisions is already the
        track-level (not node-level) parent/child structure.
        """
        children_of_track = {}
        for d in self.divisions:
            children_of_track.setdefault(d["parent_track_id"], []).extend(d["children_track_ids"])

        result_ids = {track_id} if include_self else set()
        queue = [track_id]
        while queue:
            current = queue.pop()
            for child in children_of_track.get(current, []):
                if child not in result_ids:
                    result_ids.add(child)
                    queue.append(child)
        return result_ids

    def get_descendant_node_ids(self, track_id, include_self=True):
        """Every node belonging to track_id or any of its descendant tracks — this is what a
        'highlight all descendants' feature actually needs to color/select."""
        descendant_tracks = self.get_descendant_track_ids(track_id, include_self=include_self)
        return {n.node_id for n in self.nodes if n.track_id in descendant_tracks}


class LineageBuilder:
    """
    Builds a full lineage (nodes + edges + divisions + track IDs) from a
    per-frame sequence of already-detected (and ideally already CNN-filtered)
    cell centers.

    Parameters
    ----------
    tracker : src.tracker.HungarianTracker
        Reused as-is for the normal frame-to-frame 1-to-1 linking pass.
    division_max_distance : float, optional
        Search radius for pairing an unmatched detection with a candidate
        parent. Defaults to config.DIVISION_MAX_DISTANCE if not given.
    """

    def __init__(self, tracker, division_max_distance=None):
        self.tracker = tracker
        self.division_max_distance = (
            division_max_distance
            if division_max_distance is not None
            else config.DIVISION_MAX_DISTANCE
        )

    def build(self, frames):
        """
        frames: list of Nx2 arrays (or empty lists), one per timepoint, each
        row a (y, x) center — the same shape produced by
        CellDetector.detect_volume() / src.predict.classify_centers().

        Returns a LineageResult.
        """
        nodes = []
        edges = []
        divisions = []
        tracks = {}

        next_node_id = 0
        next_track_id = 1  # CTC convention: track/label IDs start at 1; 0 is reserved to mean "no parent"

        def new_track(start_t, parent_track_id=None):
            nonlocal next_track_id
            track_id = next_track_id
            next_track_id += 1
            tracks[track_id] = {
                "start_t": start_t,
                "end_t": start_t,
                "parent_track_id": parent_track_id,
            }
            return track_id

        if len(frames) == 0:
            return LineageResult(nodes, edges, divisions, tracks)

        # --- frame 0: every detection starts its own track ---
        prev_coords = np.asarray(frames[0]) if len(frames[0]) else np.empty((0, 2))
        prev_node_ids = []
        prev_track_ids = []
        for y, x in prev_coords:
            track_id = new_track(start_t=0)
            node = LineageNode(next_node_id, 0, float(y), float(x), track_id, parent_node_id=None)
            nodes.append(node)
            prev_node_ids.append(next_node_id)
            prev_track_ids.append(track_id)
            next_node_id += 1

        # --- frames 1..T-1 ---
        for t in range(1, len(frames)):
            curr_coords = np.asarray(frames[t]) if len(frames[t]) else np.empty((0, 2))

            # Pass 1: normal 1-to-1 linking.
            matches = self.tracker.match(prev_coords, curr_coords)
            children_of_prev = {i: [] for i in range(len(prev_coords))}
            matched_curr = set()
            for m in matches:
                children_of_prev[m["previous"]].append(m["current"])
                matched_curr.add(m["current"])

            # Pass 2: try to pair up unmatched curr detections ("orphans")
            # with a frame-t detection that still has spare parent capacity.
            orphan_indices = [j for j in range(len(curr_coords)) if j not in matched_curr]
            if orphan_indices and len(prev_coords) > 0:
                capacity = [2 - len(children_of_prev[i]) for i in range(len(prev_coords))]
                expanded_cols = []  # expanded_cols[k] = original prev index for column k
                for i, cap in enumerate(capacity):
                    expanded_cols.extend([i] * max(cap, 0))

                if expanded_cols:
                    orphan_coords = curr_coords[orphan_indices]
                    parent_coords_expanded = prev_coords[expanded_cols]
                    dist_matrix = cdist(orphan_coords, parent_coords_expanded)
                    rows, cols = linear_sum_assignment(dist_matrix)
                    for r, c in zip(rows, cols):
                        if dist_matrix[r, c] <= self.division_max_distance:
                            prev_idx = expanded_cols[c]
                            curr_idx = orphan_indices[r]
                            children_of_prev[prev_idx].append(curr_idx)
                            matched_curr.add(curr_idx)

            # Now assign node IDs / track IDs for every curr detection,
            # based on how many children each prev detection ended up with.
            curr_node_ids = [None] * len(curr_coords)
            curr_track_ids = [None] * len(curr_coords)

            for prev_idx, child_indices in children_of_prev.items():
                prev_node_id = prev_node_ids[prev_idx]
                prev_track_id = prev_track_ids[prev_idx]

                if len(child_indices) == 1:
                    # Normal continuation: same track carries over.
                    curr_idx = child_indices[0]
                    y, x = curr_coords[curr_idx]
                    node = LineageNode(next_node_id, t, float(y), float(x), prev_track_id, prev_node_id)
                    nodes.append(node)
                    edges.append((prev_node_id, next_node_id))
                    curr_node_ids[curr_idx] = next_node_id
                    curr_track_ids[curr_idx] = prev_track_id
                    tracks[prev_track_id]["end_t"] = t
                    next_node_id += 1

                elif len(child_indices) == 2:
                    # Division: parent's track ends here, both children get
                    # brand-new track IDs.
                    children_node_ids = []
                    children_track_ids = []
                    for curr_idx in child_indices:
                        y, x = curr_coords[curr_idx]
                        child_track_id = new_track(start_t=t, parent_track_id=prev_track_id)
                        node = LineageNode(next_node_id, t, float(y), float(x), child_track_id, prev_node_id)
                        nodes.append(node)
                        edges.append((prev_node_id, next_node_id))
                        curr_node_ids[curr_idx] = next_node_id
                        curr_track_ids[curr_idx] = child_track_id
                        children_node_ids.append(next_node_id)
                        children_track_ids.append(child_track_id)
                        next_node_id += 1

                    divisions.append({
                        "parent_node_id": prev_node_id,
                        "parent_track_id": prev_track_id,
                        "children_node_ids": children_node_ids,
                        "children_track_ids": children_track_ids,
                        "t": t,
                    })
                # len == 0: track simply ends here (cell left the frame,
                # died, or was missed) — nothing further to do.

            # Any curr detection still unassigned is a genuinely new
            # appearance with no plausible parent: starts its own track.
            for curr_idx in range(len(curr_coords)):
                if curr_node_ids[curr_idx] is None:
                    y, x = curr_coords[curr_idx]
                    track_id = new_track(start_t=t)
                    node = LineageNode(next_node_id, t, float(y), float(x), track_id, parent_node_id=None)
                    nodes.append(node)
                    curr_node_ids[curr_idx] = next_node_id
                    curr_track_ids[curr_idx] = track_id
                    next_node_id += 1

            prev_coords = curr_coords
            prev_node_ids = curr_node_ids
            prev_track_ids = curr_track_ids

        return LineageResult(nodes, edges, divisions, tracks)
