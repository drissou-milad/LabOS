import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.tracker import HungarianTracker
from src.lineage import LineageBuilder
from src import config
from src.analytics import (
    compute_node_speeds,
    compute_track_summary,
    compute_dataset_stats,
    plot_divisions_over_time,
    plot_cell_count_over_time,
    plot_velocity_histogram,
    plot_dashboard,
)


def _synthetic_result_with_division():
    frames = [
        np.array([[10.0, 10.0], [40.0, 40.0]]),
        np.array([[11.0, 11.0], [41.0, 41.0]]),
        np.array([[12.0, 12.0], [20.0, 20.0], [42.0, 42.0]]),
        np.array([[13.0, 13.0], [21.0, 21.0], [43.0, 43.0]]),
    ]
    tracker = HungarianTracker(max_distance=5)
    builder = LineageBuilder(tracker, division_max_distance=15)
    return builder.build(frames)


def test_speeds_default_to_pixels_per_frame_when_uncalibrated():
    assert config.PIXEL_SIZE_UM is None
    assert config.FRAME_INTERVAL_MIN is None
    result = _synthetic_result_with_division()
    speeds, units = compute_node_speeds(result)
    assert units == "px/frame"
    assert len(speeds) > 0
    assert all(s >= 0 for s in speeds.values())


def test_speeds_switch_to_real_units_when_calibrated():
    result = _synthetic_result_with_division()
    speeds_px, units_px = compute_node_speeds(result)

    original_px, original_min = config.PIXEL_SIZE_UM, config.FRAME_INTERVAL_MIN
    try:
        config.PIXEL_SIZE_UM = 0.5
        config.FRAME_INTERVAL_MIN = 2.0
        speeds_um, units_um = compute_node_speeds(result)
    finally:
        config.PIXEL_SIZE_UM = original_px
        config.FRAME_INTERVAL_MIN = original_min

    assert units_px == "px/frame"
    assert units_um == "um/min"
    # Same physical motion, different units — the ratio should match the calibration factor.
    for node_id in speeds_px:
        expected = speeds_px[node_id] * 0.5 / 2.0
        assert abs(speeds_um[node_id] - expected) < 1e-9


def test_division_jump_is_included_in_speed_not_silently_dropped():
    """A daughter's first speed value is measured from the parent, not skipped — see
    compute_node_speeds' docstring for why that's a deliberate choice."""
    result = _synthetic_result_with_division()
    speeds, _ = compute_node_speeds(result)
    division = result.divisions[0]
    child_id = division["children_node_ids"][0]
    assert child_id in speeds
    assert speeds[child_id] > 0


def test_track_summary_has_expected_columns_and_row_count():
    result = _synthetic_result_with_division()
    df = compute_track_summary(result)
    assert len(df) == len(result.tracks)
    for col in ["track_id", "parent_track_id", "children_track_ids", "birth_frame",
                "death_frame", "track_length", "avg_speed"]:
        assert col in df.columns


def test_track_summary_marks_division_parent_and_children_correctly():
    result = _synthetic_result_with_division()
    df = compute_track_summary(result)
    division = result.divisions[0]
    parent_row = df[df["track_id"] == division["parent_track_id"]].iloc[0]
    assert set(parent_row["children_track_ids"]) == set(division["children_track_ids"])
    for child_track_id in division["children_track_ids"]:
        child_row = df[df["track_id"] == child_track_id].iloc[0]
        assert child_row["parent_track_id"] == division["parent_track_id"]


def test_dataset_stats_reports_correct_units_and_counts():
    result = _synthetic_result_with_division()
    stats = compute_dataset_stats(result)
    assert stats["n_cells_detected"] == len(result.nodes)
    assert stats["n_tracks"] == len(result.tracks)
    assert stats["n_divisions"] == result.n_divisions()
    assert stats["avg_lifetime_units"] == "frames"  # no calibration set
    assert stats["avg_velocity_units"] == "px/frame"


def test_explicit_calibration_overrides_global_config():
    """The actual point of the refactor: passing pixel_size_um/frame_interval_min explicitly
    must be respected even when config.py's globals say something else (or nothing)."""
    assert config.PIXEL_SIZE_UM is None  # global default, unset
    result = _synthetic_result_with_division()
    speeds, units = compute_node_speeds(result, pixel_size_um=0.5, frame_interval_min=2.0)
    assert units == "um/min"
    assert len(speeds) > 0


def test_two_different_explicit_calibrations_do_not_leak_into_each_other():
    """Regression test for the real bug found while wiring this into mvp/pipeline.py: two
    'experiments' with different Configurations, computed back-to-back (simulating what two
    background threads might each compute), must not affect each other's result — this only
    works because calibration is threaded through as explicit arguments, not read from
    config.py's global, mutable module state."""
    result = _synthetic_result_with_division()

    speeds_a, units_a = compute_node_speeds(result, pixel_size_um=1.0, frame_interval_min=1.0)
    speeds_b, units_b = compute_node_speeds(result, pixel_size_um=10.0, frame_interval_min=1.0)

    assert units_a == units_b == "um/min"
    for node_id in speeds_a:
        if speeds_a[node_id] > 0:
            assert abs(speeds_b[node_id] - speeds_a[node_id] * 10.0) < 1e-9

    # and a third call with no calibration at all, right after, should NOT have been
    # contaminated by either of the calls above (proving there's no shared mutable state)
    speeds_c, units_c = compute_node_speeds(result)
    assert units_c == "px/frame"


def test_compute_dataset_stats_respects_explicit_calibration():
    result = _synthetic_result_with_division()
    stats = compute_dataset_stats(result, pixel_size_um=2.0, frame_interval_min=5.0)
    assert stats["avg_velocity_units"] == "um/min"
    assert stats["avg_lifetime_units"] == "min"

    # same result, no calibration, computed right after — must be back to pixels/frame,
    # not contaminated by the calibrated call above
    stats_uncalibrated = compute_dataset_stats(result)
    assert stats_uncalibrated["avg_velocity_units"] == "px/frame"
    assert stats_uncalibrated["avg_lifetime_units"] == "frames"


def test_dataset_stats_handles_empty_result():
    from src.lineage import LineageResult
    empty = LineageResult([], [], [], {})
    stats = compute_dataset_stats(empty)
    assert stats["n_cells_detected"] == 0
    assert stats["avg_lifetime"] == 0.0
    assert stats["avg_velocity"] == 0.0


def test_charts_do_not_crash_and_draw_something():
    result = _synthetic_result_with_division()
    ax1 = plot_divisions_over_time(result)
    assert len(ax1.patches) > 0  # histogram bars drawn

    ax2 = plot_cell_count_over_time(result)
    assert len(ax2.lines) > 0

    speeds, units = compute_node_speeds(result)
    ax3 = plot_velocity_histogram(speeds, units=units)
    assert ax3 is not None

    axes = plot_dashboard(result)
    assert len(axes) == 3


def test_charts_handle_zero_divisions_without_crashing():
    frames = [np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]), np.array([[2.0, 2.0]])]
    result = LineageBuilder(HungarianTracker(max_distance=5)).build(frames)
    assert result.n_divisions() == 0
    ax = plot_divisions_over_time(result)  # should not raise even with an empty histogram
    assert ax is not None
