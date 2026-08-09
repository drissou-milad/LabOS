"""
Turns a LineageResult into the numbers a dashboard or a report actually wants: per-node speed,
per-track summaries (birth/death/length/parent/children/avg speed), dataset-level stats, and
the charts Phase 3 asked for (divisions over time, cell count over time, velocity histogram).

Unit honesty: every distance/time computation here defaults to pixels/frame. It only reports
real units (um, um/min) when config.PIXEL_SIZE_UM and config.FRAME_INTERVAL_MIN are both set —
see src/config.py's comment on why those default to None instead of a guessed value. Every
function that returns a speed also returns which units it's actually in, so a caller (or a
report) can't accidentally label a pixels/frame number "um/min".
"""

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from src import config


def _resolve_calibration(pixel_size_um=None, frame_interval_min=None):
    """
    Resolves the calibration to actually use: explicit arguments win, falling back to
    config.py's globals. Returns (pixel_size, frame_interval, units) where pixel_size/
    frame_interval are always real numbers (1.0 when uncalibrated, so callers can multiply
    without a None-check) and units is the honest label for what was actually used.

    Why this exists instead of just reading config.PIXEL_SIZE_UM/FRAME_INTERVAL_MIN directly
    everywhere: those are global module state, which is fine for a single notebook run but
    becomes a real correctness bug once mvp/pipeline.py can run multiple experiments with
    different per-experiment Configurations (see LabOS_Product_Spec_v1.md) — two experiments
    processing concurrently in different background threads could otherwise read each other's
    calibration. Passing explicit values through from the caller's experiment-specific
    Configuration avoids that; the global fallback stays for notebooks/scripts that don't have
    a per-experiment configuration to pass.
    """
    px = pixel_size_um if pixel_size_um is not None else config.PIXEL_SIZE_UM
    interval = frame_interval_min if frame_interval_min is not None else config.FRAME_INTERVAL_MIN
    calibrated = px is not None and interval is not None
    return (px if calibrated else 1.0), (interval if calibrated else 1.0), ("um/min" if calibrated else "px/frame")


def _units():
    _, _, units = _resolve_calibration()
    return units


def compute_node_speeds(result, pixel_size_um=None, frame_interval_min=None):
    """
    Returns dict: node_id -> speed (distance moved from the previous node in the same track,
    divided by the time between them), plus the units string it's in.

    pixel_size_um / frame_interval_min: pass an experiment's actual Configuration values here
    explicitly. If omitted, falls back to config.py's global defaults (see
    _resolve_calibration's docstring for why explicit is preferred).

    A node with no predecessor in its track (the track's first node, or a node born from a
    division — its "speed" would be measured from the parent, which is a judgment call this
    function makes explicitly: division jumps ARE included, since the daughter's position
    right after division is itself a real, meaningful displacement, not an artifact.
    """
    nodes_by_track = defaultdict(list)
    for n in result.nodes:
        nodes_by_track[n.track_id].append(n)

    node_by_id = {n.node_id: n for n in result.nodes}
    parent_of_track_start = {}
    for d in result.divisions:
        for child_track_id in d["children_track_ids"]:
            parent_of_track_start[child_track_id] = d["parent_node_id"]

    speeds = {}
    px_size, frame_interval, units = _resolve_calibration(pixel_size_um, frame_interval_min)

    for track_id, track_nodes in nodes_by_track.items():
        track_nodes = sorted(track_nodes, key=lambda n: n.t)
        prev = None
        if track_id in parent_of_track_start:
            prev = node_by_id.get(parent_of_track_start[track_id])
        for n in track_nodes:
            if prev is not None and n.t != prev.t:
                dist_px = float(np.hypot(n.y - prev.y, n.x - prev.x))
                dt = n.t - prev.t
                speeds[n.node_id] = (dist_px * px_size) / (dt * frame_interval)
            prev = n

    return speeds, units


def compute_track_summary(result, speeds=None):
    """
    One row per track: track_id, parent_track_id, children_track_ids, birth_frame, death_frame,
    track_length (frames), avg_speed. Returns a pandas DataFrame.
    """
    import pandas as pd

    if speeds is None:
        speeds, _ = compute_node_speeds(result)

    children_of_track = defaultdict(list)
    for d in result.divisions:
        children_of_track[d["parent_track_id"]].extend(d["children_track_ids"])

    node_ids_by_track = defaultdict(list)
    for n in result.nodes:
        node_ids_by_track[n.track_id].append(n.node_id)

    rows = []
    for track_id, info in result.tracks.items():
        track_speeds = [speeds[nid] for nid in node_ids_by_track[track_id] if nid in speeds]
        rows.append({
            "track_id": track_id,
            "parent_track_id": info["parent_track_id"],
            "children_track_ids": children_of_track.get(track_id, []),
            "birth_frame": info["start_t"],
            "death_frame": info["end_t"],
            "track_length": info["end_t"] - info["start_t"] + 1,
            "avg_speed": float(np.mean(track_speeds)) if track_speeds else None,
        })
    return pd.DataFrame(rows)


def compute_dataset_stats(result, speeds=None, pixel_size_um=None, frame_interval_min=None):
    """
    The dashboard numbers: n_cells_detected, n_tracks, n_divisions, avg_lifetime, avg_velocity
    — each tagged with its actual units, never silently assumed.

    pixel_size_um / frame_interval_min: same as compute_node_speeds — pass an experiment's
    actual Configuration values explicitly rather than relying on config.py's globals.
    """
    _, resolved_frame_interval, resolved_speed_units = _resolve_calibration(
        pixel_size_um, frame_interval_min
    )
    if speeds is None:
        speeds, speed_units = compute_node_speeds(result, pixel_size_um, frame_interval_min)
    else:
        speed_units = resolved_speed_units

    calibrated = resolved_speed_units == "um/min"
    lifetime_unit = "min" if calibrated else "frames"
    lifetime_multiplier = resolved_frame_interval if calibrated else 1.0

    track_lengths = [info["end_t"] - info["start_t"] + 1 for info in result.tracks.values()]

    return {
        "n_cells_detected": len(result.nodes),
        "n_tracks": len(result.tracks),
        "n_divisions": result.n_divisions(),
        "avg_lifetime": float(np.mean(track_lengths)) * lifetime_multiplier if track_lengths else 0.0,
        "avg_lifetime_units": lifetime_unit,
        "avg_velocity": float(np.mean(list(speeds.values()))) if speeds else 0.0,
        "avg_velocity_units": speed_units,
    }


def plot_divisions_over_time(result, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    division_ts = sorted(d["t"] for d in result.divisions)
    if division_ts:
        ax.hist(division_ts, bins=range(min(division_ts), max(division_ts) + 2), align="left",
                color="tab:orange", edgecolor="white")
    ax.set_xlabel("frame (t)")
    ax.set_ylabel("divisions")
    ax.set_title(f"Divisions over time (total: {len(division_ts)})")
    return ax


def plot_cell_count_over_time(result, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    counts_by_t = defaultdict(int)
    for n in result.nodes:
        counts_by_t[n.t] += 1
    ts = sorted(counts_by_t)
    ax.plot(ts, [counts_by_t[t] for t in ts], marker="o", color="tab:blue")
    ax.set_xlabel("frame (t)")
    ax.set_ylabel("cells detected")
    ax.set_title("Cell count over time")
    return ax


def plot_velocity_histogram(speeds, units=None, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    values = list(speeds.values())
    if values:
        ax.hist(values, bins=min(20, max(5, len(values) // 2)), color="tab:green", edgecolor="white")
    ax.set_xlabel(f"speed ({units or _units()})")
    ax.set_ylabel("count")
    ax.set_title("Velocity distribution")
    return ax


def plot_dashboard(result, out_ax_grid=None, pixel_size_um=None, frame_interval_min=None):
    """All three Phase-3 charts in one figure, for the report/dashboard."""
    if out_ax_grid is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    else:
        axes = out_ax_grid
        fig = axes[0].figure

    speeds, units = compute_node_speeds(result, pixel_size_um, frame_interval_min)
    plot_cell_count_over_time(result, ax=axes[0])
    plot_divisions_over_time(result, ax=axes[1])
    plot_velocity_histogram(speeds, units=units, ax=axes[2])
    fig.tight_layout()
    return axes
