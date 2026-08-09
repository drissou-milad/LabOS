"""
Turns a LineageResult (nodes + edges + divisions + tracks — see src/lineage.py) into the kind
of thing a reviewer actually wants to look at: colored trajectories over the raw image,
division events marked distinctly, a lineage tree, and summary statistics — instead of a CSV
they have to load into something else to understand.

No new required dependencies: matplotlib and numpy are already in requirements.txt.
"""

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _track_color(track_id, cmap=None):
    """A stable, distinct-looking color per track ID (same track = same color across every
    frame it appears in, which is the whole point — a reviewer should be able to follow one
    cell by eye)."""
    cmap = cmap or plt.get_cmap("tab20")
    return cmap(track_id % 20)


def plot_tracked_frame(projection, result, t, trail_length=5, ax=None):
    """
    One frame's max-projection with:
      - each detected cell as a dot, colored by its track ID (consistent across frames)
      - a short trajectory "trail" showing where each cell came from
      - division events marked with a star, at the parent's last position

    projection: 2D array, this frame's max-intensity projection.
    result: a LineageResult (see src/lineage.py).
    t: which frame to draw.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(projection, cmap="gray")

    nodes_by_track_up_to_t = defaultdict(list)
    for n in result.nodes:
        if t - trail_length <= n.t <= t:
            nodes_by_track_up_to_t[n.track_id].append(n)

    for track_id, track_nodes in nodes_by_track_up_to_t.items():
        track_nodes.sort(key=lambda n: n.t)
        color = _track_color(track_id)

        # Trail: a faded line through this track's recent positions.
        if len(track_nodes) > 1:
            ys = [n.y for n in track_nodes]
            xs = [n.x for n in track_nodes]
            ax.plot(xs, ys, "-", color=color, linewidth=1.5, alpha=0.6)

        current = [n for n in track_nodes if n.t == t]
        if current:
            n = current[0]
            ax.scatter([n.x], [n.y], s=25, color=color, zorder=3)

    division_nodes_at_t = {
        d["parent_node_id"]: d for d in result.divisions if d["t"] == t
    }
    for n in result.nodes:
        # Mark the parent's last position (t-1) with a star at the division frame, so the
        # split is visible right when it happens rather than only after the fact.
        for d in result.divisions:
            if d["t"] == t and n.node_id == d["parent_node_id"]:
                ax.scatter([n.x], [n.y], s=140, marker="*", color="white",
                           edgecolor="black", linewidth=0.8, zorder=4)

    ax.set_title(f"t = {t}  ({len(nodes_by_track_up_to_t)} tracks visible, "
                 f"{len(division_nodes_at_t)} division(s) this frame)")
    ax.axis("off")
    return ax


def render_tracked_frames(volume, result, out_dir, trail_length=5):
    """
    Saves one PNG per frame (frame_0000.png, frame_0001.png, ...) into out_dir, each showing
    that frame's max-projection with tracked cells, trails, and division markers — the
    reusable version of what mvp/streamlit_app.py's viewer step needs, so it isn't
    reimplemented separately there.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = int(max((n.t for n in result.nodes), default=-1)) + 1
    paths = []
    for t in range(n_frames):
        projection = np.asarray(volume[t]).max(axis=0) if volume is not None else None
        fig, ax = plt.subplots(figsize=(6, 6))
        if projection is not None:
            plot_tracked_frame(projection, result, t, trail_length=trail_length, ax=ax)
        else:
            # No raw image available (e.g. evaluating from saved tracks alone) — still show
            # the tracked points on a blank canvas rather than skip the frame.
            blank = np.zeros((1, 1))
            plot_tracked_frame(blank, result, t, trail_length=trail_length, ax=ax)
        out_path = out_dir / f"frame_{t:04d}.png"
        fig.tight_layout(pad=0)
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        paths.append(out_path)
    return paths


def _assign_lineage_y_positions(result):
    """Simple dendrogram-style layout: leaf tracks get sequential y-slots in start-time order;
    a dividing track's y is the average of its two children's — the same convention standard
    phylogenetic/lineage tree drawings use, so this reads the way a biologist expects it to."""
    children_of_track = defaultdict(list)
    for d in result.divisions:
        children_of_track[d["parent_track_id"]].extend(d["children_track_ids"])

    y_of_track = {}
    next_y = [0]

    def assign(track_id):
        children = sorted(children_of_track.get(track_id, []))
        if not children:
            y_of_track[track_id] = next_y[0]
            next_y[0] += 1
        else:
            for c in children:
                assign(c)
            y_of_track[track_id] = sum(y_of_track[c] for c in children) / len(children)

    root_tracks = sorted(
        tid for tid, info in result.tracks.items() if info["parent_track_id"] is None
    )
    for r in root_tracks:
        assign(r)

    return y_of_track


def plot_lineage_tree(result, ax=None, highlight_track_id=None):
    """
    A lineage tree: x-axis is time, each track is a horizontal segment from its start to end
    frame, divisions are drawn as a vertical connector splitting one line into two. This is the
    single figure that answers "what actually happened to this population over time" at a
    glance — the thing a CSV fundamentally can't do.

    highlight_track_id: if given, that track and every one of its descendants (per
    LineageResult.get_descendant_track_ids) are drawn bold and full-color; everything else is
    faded, so a researcher can immediately see one cell's whole lineage against the rest of the
    population — the actual point of building descendant lookup in the first place.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, max(4, len(result.tracks) * 0.35)))

    if not result.tracks:
        ax.text(0.5, 0.5, "No tracks", ha="center", va="center")
        ax.axis("off")
        return ax

    y_of_track = _assign_lineage_y_positions(result)

    highlighted = (
        result.get_descendant_track_ids(highlight_track_id) if highlight_track_id is not None
        else None
    )

    for track_id, info in result.tracks.items():
        y = y_of_track[track_id]
        color = _track_color(track_id)
        is_highlighted = highlighted is None or track_id in highlighted
        ax.plot(
            [info["start_t"], info["end_t"]], [y, y], "-",
            color=color if is_highlighted else "lightgray",
            linewidth=3.5 if (highlighted is not None and track_id in highlighted) else 2.5,
            alpha=1.0 if is_highlighted else 0.4,
            zorder=3 if is_highlighted else 1,
        )

    for d in result.divisions:
        parent_y = y_of_track[d["parent_track_id"]]
        is_highlighted = highlighted is None or d["parent_track_id"] in highlighted
        for child_track_id in d["children_track_ids"]:
            child_y = y_of_track[child_track_id]
            ax.plot(
                [d["t"], d["t"]], [parent_y, child_y], "--",
                color="gray" if is_highlighted else "lightgray",
                linewidth=1, alpha=1.0 if is_highlighted else 0.3,
            )

    ax.set_xlabel("frame (t)")
    ax.set_yticks([])
    title = f"Lineage tree — {len(result.tracks)} tracks, {result.n_divisions()} division(s)"
    if highlight_track_id is not None:
        title += f"  (highlighting track {highlight_track_id} + {len(highlighted) - 1} descendant(s))"
    ax.set_title(title)
    legend_elements = [
        Line2D([0], [0], linestyle="--", color="gray", label="division"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    return ax


def plot_summary_stats(result, ax=None):
    """Two-panel summary: cells detected per frame, and cumulative divisions over time — the
    kind of at-a-glance numbers a reviewer or a lab actually wants, without opening the CSV."""
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    else:
        axes = ax
        fig = axes[0].figure

    counts_by_t = defaultdict(int)
    for n in result.nodes:
        counts_by_t[n.t] += 1
    ts = sorted(counts_by_t)
    axes[0].plot(ts, [counts_by_t[t] for t in ts], marker="o")
    axes[0].set_xlabel("frame (t)")
    axes[0].set_ylabel("cells detected")
    axes[0].set_title("Cells per frame")

    division_ts = sorted(d["t"] for d in result.divisions)
    cumulative = list(range(1, len(division_ts) + 1))
    if division_ts:
        axes[1].step(division_ts, cumulative, where="post", marker="o")
    axes[1].set_xlabel("frame (t)")
    axes[1].set_ylabel("cumulative divisions")
    axes[1].set_title(f"Divisions over time (total: {len(division_ts)})")

    fig.tight_layout()
    return axes
