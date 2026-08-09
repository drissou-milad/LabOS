import numpy as np

from src.tracker import HungarianTracker
from src.lineage import LineageBuilder
from src.interactive_viewer import build_hover_data, format_hover_text


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


def test_build_hover_data_returns_only_nodes_at_the_requested_frame():
    result = _synthetic_result_with_division()
    rows = build_hover_data(result, t=2)
    assert all(True for _ in rows)  # sanity: doesn't crash
    node_ts = {n.t: n.node_id for n in result.nodes}
    returned_ids = {r["node_id"] for r in rows}
    expected_ids = {n.node_id for n in result.nodes if n.t == 2}
    assert returned_ids == expected_ids


def test_build_hover_data_includes_all_required_fields():
    result = _synthetic_result_with_division()
    rows = build_hover_data(result, t=0)
    required = {"node_id", "track_id", "y", "x", "parent_node_id", "parent_track_id",
                "children_track_ids", "birth_frame", "death_frame", "track_length",
                "speed", "speed_units", "is_division"}
    for r in rows:
        assert required <= set(r.keys())


def test_dividing_parent_is_flagged_and_shows_children():
    result = _synthetic_result_with_division()
    division = result.divisions[0]
    rows = build_hover_data(result, t=division["t"] - 1)
    parent_row = next(r for r in rows if r["node_id"] == division["parent_node_id"])
    assert parent_row["is_division"] is True

    children_row_track_ids = set()
    for t_check in range(division["t"], division["t"] + 2):
        for r in build_hover_data(result, t_check):
            if r["track_id"] in division["children_track_ids"]:
                children_row_track_ids.add(r["track_id"])
    assert children_row_track_ids == set(division["children_track_ids"])


def test_daughter_shows_correct_parent_track_and_birth_frame():
    result = _synthetic_result_with_division()
    division = result.divisions[0]
    rows = build_hover_data(result, t=division["t"])
    for child_track_id in division["children_track_ids"]:
        row = next(r for r in rows if r["track_id"] == child_track_id)
        assert row["parent_track_id"] == division["parent_track_id"]
        assert row["birth_frame"] == division["t"]


def test_format_hover_text_handles_missing_speed():
    row = {
        "node_id": 1, "track_id": 1, "parent_node_id": None, "parent_track_id": None,
        "children_track_ids": [], "birth_frame": 0, "death_frame": 3, "track_length": 4,
        "speed": None, "speed_units": "px/frame", "is_division": False,
    }
    text = format_hover_text(row)
    assert "n/a" in text
    assert "ID: 1" in text


def test_format_hover_text_shows_division_marker():
    row = {
        "node_id": 1, "track_id": 1, "parent_node_id": None, "parent_track_id": None,
        "children_track_ids": [2, 3], "birth_frame": 0, "death_frame": 3, "track_length": 4,
        "speed": 1.5, "speed_units": "px/frame", "is_division": True,
    }
    text = format_hover_text(row)
    assert "divides this frame" in text
    assert "2, 3" in text


def test_build_hover_data_empty_frame_returns_empty_list():
    result = _synthetic_result_with_division()
    rows = build_hover_data(result, t=999)  # frame with no nodes
    assert rows == []
