import numpy as np

from src.tracker import HungarianTracker
from src.lineage import LineageBuilder, LineageResult


def test_no_division_simple_continuation():
    """A single cell drifting across frames should stay on one track, with
    no division events."""
    frames = [
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 1.0]]),
        np.array([[2.0, 2.0]]),
    ]
    builder = LineageBuilder(HungarianTracker(max_distance=5), division_max_distance=10)
    result = builder.build(frames)

    assert len(result.nodes) == 3
    assert len(result.divisions) == 0
    track_ids = {n.track_id for n in result.nodes}
    assert len(track_ids) == 1


def test_track_ids_never_zero():
    """Track ID 0 is reserved by CTC convention to mean 'no parent' in the
    parent_track_id column — it must never also be a real track's own ID,
    or the two meanings become indistinguishable."""
    frames = [
        np.array([[0.0, 0.0], [50.0, 50.0]]),
        np.array([[1.0, 1.0], [51.0, 51.0]]),
    ]
    builder = LineageBuilder(HungarianTracker(max_distance=5), division_max_distance=10)
    result = builder.build(frames)

    assert 0 not in result.tracks
    ctc = result.to_ctc_tracks()
    assert 0 not in set(ctc["track_id"])


def test_division_one_child_matches_directly():
    """Parent survives as one child via normal matching; the second
    daughter is an orphan just outside tracker.max_distance but inside
    division_max_distance. Both should be recognized as one division."""
    frames = [
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 1.0]]),
        np.array([[2.0, 2.0], [12.0, 12.0]]),
    ]
    tracker = HungarianTracker(max_distance=5)
    builder = LineageBuilder(tracker, division_max_distance=20)
    result = builder.build(frames)

    assert len(result.divisions) == 1
    div = result.divisions[0]
    assert div["t"] == 2
    assert len(div["children_node_ids"]) == 2
    # the dividing parent's track should end the frame before the division
    parent_track = result.tracks[div["parent_track_id"]]
    assert parent_track["end_t"] == 1
    # both children are brand-new tracks pointing back at the parent
    for child_track_id in div["children_track_ids"]:
        assert result.tracks[child_track_id]["parent_track_id"] == div["parent_track_id"]
        assert result.tracks[child_track_id]["start_t"] == 2


def test_division_neither_child_matches_directly():
    """Both daughters land outside tracker.max_distance from the parent (a
    'full' division with no direct successor) but within
    division_max_distance — this must still be detected as a division, not
    as two unrelated new tracks."""
    frames = [
        np.array([[0.0, 0.0]]),
        np.array([[15.0, 15.0], [15.0, -15.0]]),
    ]
    tracker = HungarianTracker(max_distance=5)
    builder = LineageBuilder(tracker, division_max_distance=25)
    result = builder.build(frames)

    assert len(result.divisions) == 1
    assert len(result.divisions[0]["children_node_ids"]) == 2


def test_daughters_too_far_are_not_a_division():
    """If neither daughter is within division_max_distance either, they
    should be treated as two unrelated new appearances, not a division."""
    frames = [
        np.array([[0.0, 0.0]]),
        np.array([[100.0, 100.0], [100.0, -100.0]]),
    ]
    tracker = HungarianTracker(max_distance=5)
    builder = LineageBuilder(tracker, division_max_distance=10)
    result = builder.build(frames)

    assert len(result.divisions) == 0
    # three independent tracks: the original (now ended) plus two new ones
    assert len(result.tracks) == 3


def test_a_parent_can_have_at_most_two_children():
    """Even if three detections are all within range of one parent, at
    most 2 can be assigned as its children (a cell divides into 2, not 3);
    the third should start its own track instead of being silently dropped
    or mis-assigned."""
    frames = [
        np.array([[0.0, 0.0]]),
        np.array([[8.0, 8.0], [8.0, -8.0], [0.0, 11.0]]),
    ]
    tracker = HungarianTracker(max_distance=3)
    builder = LineageBuilder(tracker, division_max_distance=15)
    result = builder.build(frames)

    assert len(result.divisions) == 1
    assert len(result.divisions[0]["children_node_ids"]) == 2
    # total nodes at t=1: 3 detections in, 3 nodes out (2 children + 1 new track), none dropped
    t1_nodes = [n for n in result.nodes if n.t == 1]
    assert len(t1_nodes) == 3


def test_track_end_when_cell_disappears():
    """A cell that has no match in the next frame (left the field of view,
    died, or was missed) should simply end its track, not error."""
    frames = [
        np.array([[0.0, 0.0], [50.0, 50.0]]),
        np.array([[1.0, 1.0]]),  # second cell has vanished
    ]
    builder = LineageBuilder(HungarianTracker(max_distance=5), division_max_distance=10)
    result = builder.build(frames)

    assert len(result.tracks) == 2
    ended_tracks = [t for t in result.tracks.values() if t["end_t"] == 0]
    assert len(ended_tracks) == 1


def test_empty_input_does_not_crash():
    result = LineageBuilder(HungarianTracker()).build([])
    assert result.nodes == []
    assert result.edges == []
    assert result.divisions == []


def test_empty_frame_in_the_middle_does_not_crash():
    frames = [np.empty((0, 2)), np.array([[1.0, 1.0]]), np.empty((0, 2))]
    builder = LineageBuilder(HungarianTracker(max_distance=5), division_max_distance=10)
    result = builder.build(frames)

    assert len(result.nodes) == 1
    assert result.nodes[0].t == 1


def test_node_ids_are_globally_unique():
    frames = [
        np.array([[0.0, 0.0], [50.0, 50.0]]),
        np.array([[1.0, 1.0], [51.0, 51.0]]),
        np.array([[2.0, 2.0], [12.0, 12.0], [52.0, 52.0]]),  # first cell divides
    ]
    builder = LineageBuilder(HungarianTracker(max_distance=5), division_max_distance=20)
    result = builder.build(frames)

    node_ids = [n.node_id for n in result.nodes]
    assert len(node_ids) == len(set(node_ids))


def test_to_dataframe_and_to_ctc_tracks_shapes():
    frames = [
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 1.0]]),
        np.array([[2.0, 2.0], [12.0, 12.0]]),
    ]
    builder = LineageBuilder(HungarianTracker(max_distance=5), division_max_distance=20)
    result = builder.build(frames)

    df = result.to_dataframe()
    assert len(df) == len(result.nodes)
    assert set(["node_id", "t", "y", "x", "track_id", "parent_node_id",
                "parent_track_id", "is_division_parent"]).issubset(df.columns)
    assert df["is_division_parent"].sum() == 1

    ctc = result.to_ctc_tracks()
    assert len(ctc) == len(result.tracks)
    assert set(["track_id", "start_t", "end_t", "parent_track_id"]).issubset(ctc.columns)


def test_get_descendant_track_ids_walks_two_generations():
    """A divides into B,C; B later divides into D,E — descendants of A's track should be
    every track in the lineage, found by actually running a two-generation division, not
    just asserted from a one-level case."""
    frames = [
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 1.0]]),
        np.array([[2.0, 2.0], [12.0, -8.0]]),
        np.array([[3.0, 3.0], [13.0, -9.0]]),
        np.array([[4.0, 4.0], [14.0, -6.0], [14.0, -10.0]]),
        np.array([[5.0, 5.0], [15.0, -7.0], [15.0, -11.0]]),
    ]
    builder = LineageBuilder(HungarianTracker(max_distance=5), division_max_distance=15)
    result = builder.build(frames)
    assert result.n_divisions() == 2

    root_track = min(t for t, info in result.tracks.items() if info["parent_track_id"] is None)
    assert result.get_descendant_track_ids(root_track) == set(result.tracks.keys())
    assert result.get_descendant_node_ids(root_track) == {n.node_id for n in result.nodes}


def test_get_descendant_track_ids_leaf_track_is_just_itself():
    frames = [
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 1.0]]),
        np.array([[2.0, 2.0], [12.0, -8.0]]),
    ]
    builder = LineageBuilder(HungarianTracker(max_distance=5), division_max_distance=15)
    result = builder.build(frames)
    assert result.n_divisions() == 1

    leaf_track = next(
        tid for tid in result.tracks
        if not any(d["parent_track_id"] == tid for d in result.divisions)
    )
    assert result.get_descendant_track_ids(leaf_track) == {leaf_track}


def test_get_descendant_track_ids_can_exclude_self():
    frames = [
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 1.0]]),
        np.array([[2.0, 2.0], [12.0, -8.0]]),
    ]
    builder = LineageBuilder(HungarianTracker(max_distance=5), division_max_distance=15)
    result = builder.build(frames)
    root_track = min(t for t, info in result.tracks.items() if info["parent_track_id"] is None)

    with_self = result.get_descendant_track_ids(root_track, include_self=True)
    without_self = result.get_descendant_track_ids(root_track, include_self=False)
    assert root_track in with_self
    assert root_track not in without_self
    assert with_self - without_self == {root_track}


def test_from_dataframe_round_trip_preserves_everything():
    frames = [
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 1.0]]),
        np.array([[2.0, 2.0], [12.0, -8.0]]),
        np.array([[3.0, 3.0], [13.0, -9.0]]),
        np.array([[4.0, 4.0], [14.0, -6.0], [14.0, -10.0]]),
        np.array([[5.0, 5.0], [15.0, -7.0], [15.0, -11.0]]),
    ]
    from src.lineage import LineageResult
    original = LineageBuilder(HungarianTracker(max_distance=5), division_max_distance=15).build(frames)

    reconstructed = LineageResult.from_dataframe(original.to_dataframe())

    assert len(reconstructed.nodes) == len(original.nodes)
    assert sorted(reconstructed.edges) == sorted(original.edges)
    assert reconstructed.tracks == original.tracks
    assert reconstructed.n_divisions() == original.n_divisions()

    root = min(t for t, info in reconstructed.tracks.items() if info["parent_track_id"] is None)
    assert reconstructed.get_descendant_track_ids(root) == original.get_descendant_track_ids(root)


def test_from_dataframe_round_trip_through_a_real_csv_file(tmp_path):
    """The actual persistence path: write to CSV, read it back with pandas.read_csv (which
    turns missing parent_node_id into NaN, not None) — not just an in-memory DataFrame."""
    import pandas as pd
    from src.lineage import LineageResult

    frames = [
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 1.0]]),
        np.array([[2.0, 2.0], [12.0, -8.0]]),
    ]
    original = LineageBuilder(HungarianTracker(max_distance=5), division_max_distance=15).build(frames)
    csv_path = tmp_path / "tracks.csv"
    original.to_dataframe().to_csv(csv_path, index=False)

    reconstructed = LineageResult.from_dataframe(pd.read_csv(csv_path))
    assert reconstructed.n_divisions() == original.n_divisions() == 1
    assert reconstructed.tracks == original.tracks


def test_from_dataframe_handles_no_divisions():
    from src.lineage import LineageResult
    frames = [np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]), np.array([[2.0, 2.0]])]
    original = LineageBuilder(HungarianTracker(max_distance=5)).build(frames)
    reconstructed = LineageResult.from_dataframe(original.to_dataframe())
    assert reconstructed.n_divisions() == 0
    assert len(reconstructed.nodes) == 3


def test_to_dataframe_with_zero_nodes_still_has_proper_columns():
    """Regression test for a real bug: pd.DataFrame([]) has zero columns, not just zero rows —
    writing that to CSV produces a file pandas can't even read back (EmptyDataError). Found
    while running a Configuration with an absurdly strict detection threshold that legitimately
    detected nothing at all."""
    import pandas as pd

    empty = LineageResult([], [], [], {})
    df = empty.to_dataframe()
    assert list(df.columns) == ["node_id", "t", "y", "x", "track_id", "parent_node_id",
                                 "parent_track_id", "is_division_parent"]
    assert len(df) == 0

    # the actual failure mode: write to CSV, read it back
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "tracks.csv")
        df.to_csv(csv_path, index=False)
        reloaded = pd.read_csv(csv_path)  # must not raise EmptyDataError
        assert list(reloaded.columns) == list(df.columns)
        assert len(reloaded) == 0


def test_to_ctc_tracks_with_zero_tracks_still_has_proper_columns():
    import pandas as pd
    empty = LineageResult([], [], [], {})
    df = empty.to_ctc_tracks()
    assert list(df.columns) == ["track_id", "start_t", "end_t", "parent_track_id"]

    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "ctc_tracks.csv")
        df.to_csv(csv_path, index=False)
        reloaded = pd.read_csv(csv_path)
        assert list(reloaded.columns) == list(df.columns)


def test_from_dataframe_handles_a_zero_node_round_trip():
    """The full chain the original bug broke: build an empty result, export, write to disk,
    read back, reconstruct — must not raise anywhere in that chain."""
    import pandas as pd
    import tempfile, os

    empty = LineageResult([], [], [], {})
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "tracks.csv")
        empty.to_dataframe().to_csv(csv_path, index=False)
        reloaded_df = pd.read_csv(csv_path)
        reconstructed = LineageResult.from_dataframe(reloaded_df)
        assert len(reconstructed.nodes) == 0
        assert reconstructed.n_divisions() == 0


def test_division_max_distance_defaults_from_config_when_not_given():
    from src import config
    builder = LineageBuilder(HungarianTracker(max_distance=5))
    assert builder.division_max_distance == config.DIVISION_MAX_DISTANCE
