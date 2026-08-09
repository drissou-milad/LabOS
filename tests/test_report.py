import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.tracker import HungarianTracker
from src.lineage import LineageBuilder
from src.report import generate_report


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


def test_generate_report_without_volume_still_produces_core_files(tmp_path):
    result, frames = _synthetic_result_with_division()
    paths = generate_report(result, tmp_path, volume=None, sample_name="no_volume_test")

    assert paths["tracks_csv"].exists()
    assert paths["ctc_tracks_csv"].exists()
    assert paths["lineage_tree_png"].exists()
    assert paths["summary_stats_png"].exists()
    assert paths["report_pdf"].exists()
    assert paths["report_pdf"].stat().st_size > 500
    assert "frames" not in paths


def test_generate_report_with_volume_includes_frames(tmp_path):
    result, frames = _synthetic_result_with_division()
    volume = [np.random.rand(1, 64, 64) for _ in frames]
    paths = generate_report(result, tmp_path, volume=volume, sample_name="with_volume_test")

    assert "frames" in paths
    assert len(paths["frames"]) == len(frames)


def test_generate_report_embeds_scores_when_given(tmp_path):
    result, frames = _synthetic_result_with_division()
    scores = {"adjusted_edge_jaccard": 0.9, "division_jaccard": 1.0, "final_score": 1.0}
    paths = generate_report(result, tmp_path, volume=None, scores=scores, sample_name="scored_test")

    import json
    summary = json.loads(paths["summary_json"].read_text())
    assert summary["scores"] == scores


def test_generate_report_with_benchmark_rows_adds_pages(tmp_path):
    from src.benchmark import score_external_method

    result, frames = _synthetic_result_with_division()
    pred_df = result.to_dataframe()
    gt_nodes = pred_df[["node_id", "t", "y", "x"]].to_dict("records")
    row = score_external_method("test", "self-match", gt_nodes, result.edges, gt_nodes, result.edges,
                                 n_true_nodes_estimate=len(gt_nodes), max_distance=2.0)

    paths_without = generate_report(result, tmp_path / "no_bench", volume=None, sample_name="t")
    paths_with = generate_report(result, tmp_path / "with_bench", volume=None, sample_name="t",
                                  benchmark_rows=[row])

    import pypdf
    pages_without = len(pypdf.PdfReader(str(paths_without["report_pdf"])).pages)
    pages_with = len(pypdf.PdfReader(str(paths_with["report_pdf"])).pages)
    assert pages_with > pages_without


def test_division_analysis_text_lists_real_divisions():
    from src.report import _division_analysis_text
    result, frames = _synthetic_result_with_division()
    text = _division_analysis_text(result)
    division = result.divisions[0]
    assert f"track {division['parent_track_id']}" in text
    for child in division["children_track_ids"]:
        assert str(child) in text


def test_division_analysis_text_handles_no_divisions():
    from src.lineage import LineageResult
    from src.report import _division_analysis_text
    empty = LineageResult([], [], [], {})
    text = _division_analysis_text(empty)
    assert "No divisions detected" in text


def test_appendix_text_includes_real_config_values():
    from src.report import _appendix_text
    from src import config
    result, frames = _synthetic_result_with_division()
    text = _appendix_text(result)
    assert f"DETECTION_THRESHOLD = {config.DETECTION_THRESHOLD}" in text
    assert "Track Summary" in text


def test_limitations_text_flags_missing_calibration():
    from src.report import _limitations_text
    from src import config
    assert config.PIXEL_SIZE_UM is None  # default state
    text = _limitations_text()
    assert "NOT set" in text
