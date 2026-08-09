"""
The interactive per-frame viewer: click/hover a cell, see its ID, parent, children, birth
frame, death frame, track length, and speed — plus real zoom/pan, via Plotly (which gives both
of those essentially for free through its default toolbar, rather than requiring a bespoke
canvas frontend).

Split deliberately into two layers:

1. build_hover_data() — pure Python/numpy, no Plotly import. Fully tested in this environment.
2. build_plotly_figure() — the actual Plotly rendering. Plotly is NOT installed in the
   environment that wrote this (no PyPI access to install it), so this function is written
   carefully against Plotly's documented graph_objects API but has never actually been run.
   Same honesty standard as src/benchmark.py's load_trackmate_xml(): verify it renders before
   trusting it in front of anyone. Add "plotly" to requirements.txt before running it.

Usage in Streamlit: `st.plotly_chart(build_plotly_figure(...), use_container_width=True)`
"""

from collections import defaultdict

from src.analytics import compute_node_speeds


def build_hover_data(result, t, speeds=None, speed_units=None):
    """
    Returns a list of dicts, one per cell visible at frame t, each with everything the Phase-2
    spec asked a click/hover to show:
        {node_id, track_id, y, x, parent_node_id, parent_track_id, children_track_ids,
         birth_frame, death_frame, track_length, speed, speed_units, is_division}

    No Plotly dependency — this is what test_interactive_viewer.py actually exercises.
    """
    if speeds is None:
        speeds, speed_units = compute_node_speeds(result)
    speed_units = speed_units or "px/frame"

    children_of_track = defaultdict(list)
    for d in result.divisions:
        children_of_track[d["parent_track_id"]].extend(d["children_track_ids"])
    dividing_parent_node_ids = {d["parent_node_id"] for d in result.divisions}

    rows = []
    for n in result.nodes:
        if n.t != t:
            continue
        track_info = result.tracks[n.track_id]
        rows.append({
            "node_id": n.node_id,
            "track_id": n.track_id,
            "y": n.y,
            "x": n.x,
            "parent_node_id": n.parent_node_id,
            "parent_track_id": track_info["parent_track_id"],
            "children_track_ids": children_of_track.get(n.track_id, []),
            "birth_frame": track_info["start_t"],
            "death_frame": track_info["end_t"],
            "track_length": track_info["end_t"] - track_info["start_t"] + 1,
            "speed": speeds.get(n.node_id),
            "speed_units": speed_units,
            "is_division": n.node_id in dividing_parent_node_ids,
        })
    return rows


def format_hover_text(row):
    """One cell's hover_data dict -> the multi-line text Plotly's hovertemplate/hovertext
    shows. Separated from build_hover_data so the text formatting itself is testable without
    needing a full LineageResult in every test."""
    speed_str = f"{row['speed']:.3f} {row['speed_units']}" if row["speed"] is not None else "n/a"
    children_str = ", ".join(str(c) for c in row["children_track_ids"]) or "none"
    lines = [
        f"ID: {row['node_id']} (track {row['track_id']})",
        f"Parent track: {row['parent_track_id'] if row['parent_track_id'] is not None else 'none'}",
        f"Children tracks: {children_str}",
        f"Birth frame: {row['birth_frame']}",
        f"Death frame: {row['death_frame']}",
        f"Track length: {row['track_length']} frames",
        f"Speed: {speed_str}",
    ]
    if row["is_division"]:
        lines.append("(divides this frame)")
    return "<br>".join(lines)


def build_plotly_figure(projection, result, t, speeds=None, speed_units=None, trail_length=5,
                         highlight_track_id=None):
    """
    NOT EXECUTED IN THE ENVIRONMENT THAT WROTE THIS — plotly isn't installed here. Written
    against Plotly's documented graph_objects API. Verify this actually renders before trusting
    it — see this module's docstring.

    highlight_track_id: if given, that track and all its descendants (per
    LineageResult.get_descendant_track_ids) are drawn full-opacity and larger; every other cell
    is faded — "highlight all descendants" for the per-frame view, matching what
    src/visualize.py's plot_lineage_tree does for the lineage tree.
    """
    import plotly.graph_objects as go

    hover_rows = build_hover_data(result, t, speeds=speeds, speed_units=speed_units)
    highlighted_tracks = (
        result.get_descendant_track_ids(highlight_track_id) if highlight_track_id is not None
        else None
    )

    fig = go.Figure()

    # The image itself, as a grayscale heatmap — Plotly's Heatmap gives zoom/pan/reset for free
    # via its default toolbar, which is most of what "Phase 2: zoom, pan" was asking for.
    fig.add_trace(go.Heatmap(
        z=projection,
        colorscale="gray",
        showscale=False,
        hoverinfo="skip",
    ))

    # Trails: short lines through each visible track's recent positions.
    nodes_by_track = defaultdict(list)
    for n in result.nodes:
        if t - trail_length <= n.t <= t:
            nodes_by_track[n.track_id].append(n)
    for track_id, track_nodes in nodes_by_track.items():
        track_nodes = sorted(track_nodes, key=lambda n: n.t)
        is_highlighted = highlighted_tracks is None or track_id in highlighted_tracks
        if len(track_nodes) > 1:
            fig.add_trace(go.Scatter(
                x=[n.x for n in track_nodes],
                y=[n.y for n in track_nodes],
                mode="lines",
                line=dict(width=1.5),
                opacity=0.5 if is_highlighted else 0.1,
                hoverinfo="skip",
                showlegend=False,
            ))

    # The cells themselves — this is the "click a cell, see its info" requirement. Plotly
    # shows hover_text on mouseover by default; click-to-pin behavior needs a small amount of
    # Streamlit-side wiring (st.plotly_chart's selection events) on top of this, noted in
    # RUNBOOK.md rather than guessed at here.
    if hover_rows:
        sizes = []
        opacities = []
        for r in hover_rows:
            is_highlighted = highlighted_tracks is None or r["track_id"] in highlighted_tracks
            sizes.append(14 if is_highlighted else 8)
            opacities.append(1.0 if is_highlighted else 0.25)
        fig.add_trace(go.Scatter(
            x=[r["x"] for r in hover_rows],
            y=[r["y"] for r in hover_rows],
            mode="markers",
            marker=dict(size=sizes, color=[r["track_id"] for r in hover_rows], colorscale="Turbo",
                        opacity=opacities),
            text=[format_hover_text(r) for r in hover_rows],
            hoverinfo="text",
            showlegend=False,
        ))

    fig.update_yaxes(autorange="reversed")  # image-style (row 0 at top), not plot-style
    fig.update_layout(
        title=f"t = {t}",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, scaleanchor="x"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
