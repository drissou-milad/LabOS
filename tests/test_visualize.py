import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.tracker import HungarianTracker
from src.lineage import LineageBuilder
from src.visualize import plot_tracked_frame, render_tracked_frames, plot_lineage_tree, plot_summary_stats


def _synthetic_result_with_division():
    frames = [
        np.array([[10.0, 10.0], [40.0, 40.0]]),
        np.array([[11.0, 11.0], [41.0, 41.0]]),
        np.array([[12.0, 12.0], [20.0, 20.0], [42.0, 42.0]]),
        np.array([[13.0, 13.0], [21.0, 21.0], [43.0, 43.0]]),
    ]
    tracker = HungarianTracker(max_distance=5)
    builder = LineageBuilder(tracker, division_max_distance=15)
    return builder.build(frames), frames


def test_plot_tracked_frame_does_not_crash_and_draws_something():
    result, frames = _synthetic_result_with_division()
    projection = np.random.rand(64, 64)
    ax = plot_tracked_frame(projection, result, t=2)
    assert len(ax.collections) > 0  # scatter points were actually drawn


def test_render_tracked_frames_writes_one_file_per_frame(tmp_path):
    result, frames = _synthetic_result_with_division()
    volume = [np.random.rand(1, 64, 64) for _ in frames]
    paths = render_tracked_frames(volume, result, tmp_path)
    assert len(paths) == len(frames)
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0


def test_plot_lineage_tree_handles_empty_result():
    from src.lineage import LineageResult
    empty = LineageResult([], [], [], {})
    ax = plot_lineage_tree(empty)
    assert ax is not None  # shouldn't raise


def test_plot_lineage_tree_draws_a_division_branch():
    result, _ = _synthetic_result_with_division()
    ax = plot_lineage_tree(result)
    # a division should produce at least one dashed connector line
    dashed_lines = [line for line in ax.get_lines() if line.get_linestyle() == "--"]
    assert len(dashed_lines) >= 1


def _two_generation_result():
    frames = [
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 1.0]]),
        np.array([[2.0, 2.0], [12.0, -8.0]]),
        np.array([[3.0, 3.0], [13.0, -9.0]]),
        np.array([[4.0, 4.0], [14.0, -6.0], [14.0, -10.0]]),
        np.array([[5.0, 5.0], [15.0, -7.0], [15.0, -11.0]]),
    ]
    builder = LineageBuilder(HungarianTracker(max_distance=5), division_max_distance=15)
    return builder.build(frames)


def test_plot_lineage_tree_highlight_includes_all_descendants_in_title():
    result = _two_generation_result()
    root_track = min(t for t, info in result.tracks.items() if info["parent_track_id"] is None)
    ax = plot_lineage_tree(result, highlight_track_id=root_track)
    assert f"track {root_track}" in ax.get_title()
    assert "4 descendant" in ax.get_title()  # root + 2 children + 2 grandchildren - self = 4


def test_plot_lineage_tree_highlight_leaf_has_zero_descendants():
    result = _two_generation_result()
    leaf = next(
        tid for tid in result.tracks
        if not any(d["parent_track_id"] == tid for d in result.divisions)
    )
    ax = plot_lineage_tree(result, highlight_track_id=leaf)
    assert "0 descendant" in ax.get_title()


def test_plot_lineage_tree_without_highlight_still_works():
    """Backward compatibility: highlight_track_id is optional."""
    result = _two_generation_result()
    ax = plot_lineage_tree(result)
    assert "highlighting" not in ax.get_title()


def test_plot_summary_stats_covers_all_frames():
    result, frames = _synthetic_result_with_division()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2)
    plot_summary_stats(result, ax=axes)
    xdata = axes[0].lines[0].get_xdata()
    assert len(xdata) == len(frames)
