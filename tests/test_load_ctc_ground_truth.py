"""
Tests scripts/load_ctc_ground_truth.py against tests/fixtures/ctc_sample/ — a hand-authored
CTC-format fixture (see that module's docstring: NOT a real downloaded CTC dataset, since this
sandbox has no network access). These tests prove the loader correctly implements the CTC
schema it claims to; they don't prove it survives a real dataset's edge cases (missing frames,
gaps in tracks, larger label counts). Re-run this kind of check against a real download before
trusting a comparison table built from it — see RUNBOOK.md's CTC step.
"""

from pathlib import Path

from scripts.load_ctc_ground_truth import load_ctc_ground_truth
from src.benchmark import score_method

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "ctc_sample" / "01_GT" / "TRA"


def test_load_ctc_ground_truth_parses_all_label_instances_as_nodes():
    nodes, edges = load_ctc_ground_truth(FIXTURE_DIR)
    assert len(nodes) == 7  # labels 1,2 in frames 0-1, labels 2,3,4 in frame 2
    node = next(n for n in nodes if n["node_id"] == "1_0")
    assert node == {"node_id": "1_0", "t": 0, "y": 1.5, "x": 1.5}


def test_load_ctc_ground_truth_computes_real_centroids_not_bounding_box_corners():
    """The fixture's label 3 occupies rows/cols 0-1 (a 2x2 block) — its centroid is (0.5, 0.5),
    not (0, 0) or (1, 1); this catches a loader that used e.g. bounding-box min/max instead of
    an actual pixel-mean centroid."""
    nodes, edges = load_ctc_ground_truth(FIXTURE_DIR)
    daughter = next(n for n in nodes if n["node_id"] == "3_2")
    assert daughter["y"] == 0.5 and daughter["x"] == 0.5


def test_load_ctc_ground_truth_builds_within_track_edges():
    nodes, edges = load_ctc_ground_truth(FIXTURE_DIR)
    assert ("1_0", "1_1") in edges  # track 1, frame 0 -> frame 1
    assert ("2_0", "2_1") in edges and ("2_1", "2_2") in edges  # track 2 runs all 3 frames


def test_load_ctc_ground_truth_builds_division_edges_from_man_track_txt():
    """man_track.txt's parent column (not the mask images) is what encodes a division — track
    1 ends at frame 1, tracks 3 and 4 begin at frame 2 with parent=1."""
    nodes, edges = load_ctc_ground_truth(FIXTURE_DIR)
    assert ("1_1", "3_2") in edges
    assert ("1_1", "4_2") in edges
    # track 1's own label never appears again after its own last frame
    assert not any(src == "1_2" or tgt == "1_2" for src, tgt in edges)


def test_load_ctc_ground_truth_result_scores_perfectly_against_itself():
    nodes, edges = load_ctc_ground_truth(FIXTURE_DIR)
    scores = score_method(nodes, edges, nodes, edges, n_true_nodes_estimate=len(nodes), max_distance=2.0)
    assert scores["edge_precision"] == 1.0
    assert scores["edge_recall"] == 1.0
    assert scores["division_jaccard"] == 1.0


def test_load_ctc_ground_truth_raises_a_clear_error_for_a_missing_folder(tmp_path):
    try:
        load_ctc_ground_truth(tmp_path / "does_not_exist")
        assert False, "expected a FileNotFoundError"
    except FileNotFoundError as e:
        assert "man_track.txt" in str(e)
