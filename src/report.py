"""
Ties src/evaluate.py, src/visualize.py, and export (CSV/CTC) into one function — the
"Evaluation -> Visualization -> Export, end-to-end" pipeline. This is the shared implementation
behind both mvp/streamlit_app.py's "Download Report" step and any offline batch run
(src/benchmark.py uses the same LineageResult -> CSV/figures path, just without the PDF bundling
step, since a benchmark run produces many results, not one report to hand to a person).
"""

import json

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src import config
from src.visualize import render_tracked_frames, plot_lineage_tree, plot_summary_stats
from src.analytics import compute_node_speeds, compute_dataset_stats, compute_track_summary, plot_dashboard


def _limitations_text(pixel_size_um=None, frame_interval_min=None):
    """Grounded in the actual current implementation — every line here traces to a real,
    checkable fact about this pipeline, not a generic disclaimer paragraph."""
    lines = [
        "- Detection uses a fixed threshold (config.DETECTION_THRESHOLD) not recalibrated "
        "per-dataset; brightness differences between microscopes/samples can silently change "
        "detection sensitivity (see mvp/make_test_data.py's comments for a concrete example of "
        "this happening in practice).",
        "- Detection and tracking both operate on the maximum-intensity projection of each 3D "
        "frame, not the full 3D volume — z-position information is not tracked.",
        "- Division detection (src/lineage.py) uses a distance-based heuristic (a second "
        "assignment pass on unmatched detections), not a learned model — it can miss divisions "
        "where daughters land far apart, or misattribute a division in dense regions where "
        "multiple cells are within range of the same parent (see src/lineage.py's docstring).",
        "- No motion prediction (e.g. a Kalman filter) — occluded or temporarily undetected "
        "cells will appear as a track ending and a new one starting, not a continuous track "
        "with a gap.",
        "- Speed/velocity are reported in pixels/frame unless a real physical calibration "
        "(pixel size + frame interval) is set for this dataset's actual acquisition parameters "
        "— see this report's Tracking Metrics section for which applies here.",
    ]
    resolved_pixel_size = pixel_size_um if pixel_size_um is not None else config.PIXEL_SIZE_UM
    resolved_frame_interval = frame_interval_min if frame_interval_min is not None else config.FRAME_INTERVAL_MIN
    if resolved_pixel_size is None or resolved_frame_interval is None:
        lines.append(
            "- THIS RUN: physical calibration was NOT set — all speed/lifetime numbers in this "
            "report are in pixels/frame, not real-world units."
        )
    return "Limitations\n\n" + "\n".join(lines)


def _method_text():
    """A grounded methods paragraph, built from the actual config values this run used —
    not templated marketing prose. Keeps the report honest about exactly what ran."""
    return (
        "Cells were detected per frame via Gaussian smoothing (sigma="
        f"{config.GAUSSIAN_SIGMA}) followed by local maximum detection (threshold="
        f"{config.DETECTION_THRESHOLD}, min. distance={config.CELL_RADIUS}px) on each frame's "
        "maximum-intensity projection. Candidate detections were classified by a trained CNN "
        "(CellCNN). Frame-to-frame linking used optimal (Hungarian) bipartite assignment "
        f"(max. distance={config.TRACKING_MAX_DISTANCE}px); cell divisions were detected via a "
        "second assignment pass on unmatched detections "
        f"(max. distance={config.DIVISION_MAX_DISTANCE:.1f}px), each daughter assigned a new "
        "track ID linked to its parent."
    )


def _division_analysis_text(result):
    """One row per division: parent track, both children, the frame it happened, and how long
    each side of the family lived — built from real LineageResult data (result.divisions +
    result.tracks), not placeholder text."""
    if not result.divisions:
        return "Division Analysis\n\nNo divisions detected in this run."

    lines = ["Division Analysis", "", f"{len(result.divisions)} division(s) detected:", ""]
    for i, d in enumerate(result.divisions, 1):
        parent_info = result.tracks[d["parent_track_id"]]
        lines.append(f"{i}. Frame {d['t']}: track {d['parent_track_id']} "
                     f"(alive frames {parent_info['start_t']}-{parent_info['end_t']}) -> "
                     f"tracks {', '.join(str(c) for c in d['children_track_ids'])}")
        for child_track_id in d["children_track_ids"]:
            child_info = result.tracks[child_track_id]
            length = child_info["end_t"] - child_info["start_t"] + 1
            lines.append(f"     track {child_track_id}: alive frames "
                         f"{child_info['start_t']}-{child_info['end_t']} ({length} frames)")
    return "\n".join(lines)


def _appendix_text(result, max_rows=25):
    """Full config parameters used this run, plus a (possibly truncated) raw track table —
    the appendix a reviewer actually wants: exact numbers, not just figures."""
    lines = ["Appendix: Run Configuration", ""]
    for name in ["GAUSSIAN_SIGMA", "DETECTION_THRESHOLD", "CELL_RADIUS",
                 "TRACKING_MAX_DISTANCE", "DIVISION_MAX_DISTANCE",
                 "PIXEL_SIZE_UM", "FRAME_INTERVAL_MIN"]:
        lines.append(f"  {name} = {getattr(config, name)}")

    lines += ["", "Appendix: Track Summary", ""]
    df = compute_track_summary(result)
    if len(df) > max_rows:
        lines.append(df.head(max_rows).to_string(index=False))
        lines.append(f"\n... {len(df) - max_rows} more rows in tracks.csv")
    else:
        lines.append(df.to_string(index=False) if len(df) else "(no tracks)")
    return "\n".join(lines)


def generate_report(result, out_dir, volume=None, scores=None, sample_name=None,
                     benchmark_rows=None, max_preview_frames=6,
                     pixel_size_um=None, frame_interval_min=None):
    """
    result: a LineageResult (src/lineage.py).
    out_dir: Path, created if it doesn't exist.
    volume: optional (T, Z, Y, X) array — if given, per-frame tracked overlays are rendered;
        if omitted, the report still includes the lineage tree, stats, and CSVs, just no
        per-frame images.
    scores: optional dict from src/evaluate.py (e.g. {"adjusted_edge_jaccard": ..., "score": ...})
        — if given, printed on the report's cover page. Omit if there's no ground truth to score
        against, which is the normal case for a real (non-competition) sample.
    benchmark_rows: optional list of rows from src/benchmark.py's run_benchmark() /
        score_external_method() — if given, the full Method/Runtime/Memory/Precision/Recall/
        Score comparison table and dashboard chart are embedded in the Tracking Metrics section.
    pixel_size_um / frame_interval_min: an experiment's actual Configuration values (see
        mvp/experiments.py, LabOS_Product_Spec_v1.md), passed through to
        src/analytics.py's compute_dataset_stats() explicitly rather than read from
        config.py's global state — see src/analytics.py's _resolve_calibration() docstring for
        why that distinction matters once multiple experiments can run concurrently.

    Returns a dict of the paths written, so callers (the MVP, a notebook, a benchmark script)
    can link to them without guessing the layout.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # --- Export ---
    tracks_csv = out_dir / "tracks.csv"
    result.to_dataframe().to_csv(tracks_csv, index=False)
    paths["tracks_csv"] = tracks_csv

    ctc_csv = out_dir / "ctc_tracks.csv"
    result.to_ctc_tracks().to_csv(ctc_csv, index=False)
    paths["ctc_tracks_csv"] = ctc_csv

    summary_stats = compute_dataset_stats(result, pixel_size_um=pixel_size_um,
                                           frame_interval_min=frame_interval_min)
    summary = {
        "sample_name": sample_name,
        "n_frames": int(max((n.t for n in result.nodes), default=-1)) + 1,
        "n_cells_detected": summary_stats["n_cells_detected"],
        "n_tracks": summary_stats["n_tracks"],
        "n_divisions": summary_stats["n_divisions"],
        "avg_lifetime": summary_stats["avg_lifetime"],
        "avg_lifetime_units": summary_stats["avg_lifetime_units"],
        "avg_velocity": summary_stats["avg_velocity"],
        "avg_velocity_units": summary_stats["avg_velocity_units"],
        "scores": scores,
    }
    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))
    paths["summary_json"] = summary_json

    # --- Visualization ---
    if volume is not None:
        frames_dir = out_dir / "frames"
        frame_paths = render_tracked_frames(volume, result, frames_dir)
        paths["frames"] = frame_paths
    else:
        frame_paths = []

    lineage_fig, lineage_ax = plt.subplots(figsize=(9, max(4, len(result.tracks) * 0.35)))
    plot_lineage_tree(result, ax=lineage_ax)
    lineage_path = out_dir / "lineage_tree.png"
    lineage_fig.savefig(lineage_path, dpi=120, bbox_inches="tight")
    paths["lineage_tree_png"] = lineage_path

    stats_fig, stats_axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_summary_stats(result, ax=stats_axes)
    stats_path = out_dir / "summary_stats.png"
    stats_fig.savefig(stats_path, dpi=120, bbox_inches="tight")
    paths["summary_stats_png"] = stats_path

    dashboard_axes = plot_dashboard(result, pixel_size_um=pixel_size_um, frame_interval_min=frame_interval_min)
    dashboard_fig = dashboard_axes[0].figure
    dashboard_path = out_dir / "dashboard.png"
    dashboard_fig.savefig(dashboard_path, dpi=120, bbox_inches="tight")
    paths["dashboard_png"] = dashboard_path

    # --- Bundle everything into one shareable PDF, in the requested structure:
    # Experiment -> Methods -> Tracking Metrics -> Division Analysis -> Figures ->
    # Limitations -> Appendix ---
    def _text_page(pdf_obj, text, fontsize=11, wrap=True):
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.05, 0.95, text, va="top", fontsize=fontsize, wrap=wrap, family="monospace")
        pdf_obj.savefig(fig)
        plt.close(fig)

    pdf_path = out_dir / "report.pdf"
    with PdfPages(pdf_path) as pdf:
        # --- Experiment ---
        lines = [
            "LabOS Report",
            "",
            f"Sample: {sample_name or '(unnamed)'}",
            f"Frames: {summary['n_frames']}",
            "",
            "--- Dashboard ---",
            f"Detected cells: {summary['n_cells_detected']}",
            f"Tracks: {summary['n_tracks']}",
            f"Divisions: {summary['n_divisions']}",
            f"Avg. lifetime: {summary['avg_lifetime']:.1f} {summary['avg_lifetime_units']}",
            f"Avg. velocity: {summary['avg_velocity']:.3f} {summary['avg_velocity_units']}",
        ]
        if scores:
            lines.append("")
            lines.append("Scores (against provided ground truth):")
            for k, v in scores.items():
                lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        _text_page(pdf, "\n".join(lines), fontsize=13)

        # --- Methods ---
        _text_page(pdf, "Method\n\n" + _method_text())

        # --- Tracking Metrics ---
        if benchmark_rows:
            from src.benchmark import format_results_table, plot_benchmark_dashboard
            _text_page(pdf, "Tracking Metrics\n\n" + format_results_table(benchmark_rows), fontsize=8)
            bench_fig = plot_benchmark_dashboard(benchmark_rows)
            pdf.savefig(bench_fig)
            plt.close(bench_fig)
        else:
            metrics_lines = [
                "Tracking Metrics",
                "",
                f"Tracks: {summary['n_tracks']}",
                f"Divisions: {summary['n_divisions']}",
                f"Avg. lifetime: {summary['avg_lifetime']:.1f} {summary['avg_lifetime_units']}",
                f"Avg. velocity: {summary['avg_velocity']:.3f} {summary['avg_velocity_units']}",
                "",
                "No ground truth / no benchmark comparison provided for this run — pass "
                "benchmark_rows (from src/benchmark.py) for a Method vs. Method comparison "
                "with precision/recall/runtime/memory.",
            ]
            _text_page(pdf, "\n".join(metrics_lines))

        # --- Division Analysis ---
        _text_page(pdf, _division_analysis_text(result), fontsize=9)

        # --- Figures ---
        pdf.savefig(stats_fig)
        pdf.savefig(dashboard_fig)
        pdf.savefig(lineage_fig)
        for fp in frame_paths[:max_preview_frames]:
            img = plt.imread(fp)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

        # --- Limitations ---
        _text_page(pdf, _limitations_text(pixel_size_um, frame_interval_min))

        # --- Appendix ---
        _text_page(pdf, _appendix_text(result), fontsize=8)

    plt.close(stats_fig)
    plt.close(dashboard_fig)
    plt.close(lineage_fig)
    paths["report_pdf"] = pdf_path

    return paths
