import numpy as np

from src.tracker import HungarianTracker


def test_match_simple_one_to_one():
    """Two points that each move a small, unambiguous distance should match
    to each other, not get crossed."""
    previous = np.array([[0, 0], [100, 100]])
    current = np.array([[2, 2], [103, 101]])

    tracker = HungarianTracker(max_distance=25)
    matches = tracker.match(previous, current)

    assert len(matches) == 2
    matched_pairs = {(m["previous"], m["current"]) for m in matches}
    assert (0, 0) in matched_pairs
    assert (1, 1) in matched_pairs


def test_match_respects_max_distance():
    """A current point far outside max_distance from every previous point
    should not be force-matched to the nearest one."""
    previous = np.array([[0, 0]])
    current = np.array([[1000, 1000]])

    tracker = HungarianTracker(max_distance=25)
    matches = tracker.match(previous, current)

    assert matches == []


def test_match_empty_inputs():
    tracker = HungarianTracker()
    assert tracker.match([], []) == []
    assert tracker.match(np.array([[0, 0]]), []) == []
    assert tracker.match([], np.array([[0, 0]])) == []


def test_track_across_multiple_frames():
    """A single point drifting by a small, consistent amount each frame
    should stay matched frame-to-frame."""
    detections = [
        np.array([[0, 0]]),
        np.array([[3, 3]]),
        np.array([[6, 5]]),
    ]

    tracker = HungarianTracker(max_distance=10)
    all_matches = tracker.track(detections)

    assert len(all_matches) == len(detections) - 1
    for frame_matches in all_matches:
        assert len(frame_matches) == 1
        assert frame_matches[0]["previous"] == 0
        assert frame_matches[0]["current"] == 0


def test_match_picks_globally_optimal_assignment():
    """Classic case where greedy nearest-neighbor would get it wrong but the
    Hungarian algorithm (globally optimal) gets it right."""
    previous = np.array([[0, 0], [10, 0]])
    current = np.array([[1, 0], [9, 0]])

    tracker = HungarianTracker(max_distance=25)
    matches = tracker.match(previous, current)

    matched_pairs = {(m["previous"], m["current"]) for m in matches}
    assert matched_pairs == {(0, 0), (1, 1)}
