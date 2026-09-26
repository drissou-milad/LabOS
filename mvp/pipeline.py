"""
The MVP's actual logic (upload parsing, job processing, status), separated from
mvp/streamlit_app.py's UI code so it can be tested without `streamlit` installed — and so it's
reusable if a future v1 frontend (see docs/PHASE2_MVP_ARCHITECTURE.md) wants the same pipeline
behind a different UI.

streamlit_app.py imports from this module; it should contain no `import streamlit`.
"""

import json
import sys
import zipfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.detector import CellDetector
from src.tracker import HungarianTracker
from src.lineage import LineageBuilder
from src.report import generate_report

import mvp.experiments as experiments

STORAGE_DIR = PROJECT_ROOT / "mvp" / "storage"

# In-memory cache of {job_id: {"result": LineageResult, "volume": ndarray}}, so the UI can
# build the interactive Plotly viewer (src/interactive_viewer.py) without round-tripping
# through tracks.csv. v0-scale limitation, consistent with the rest of this app: doesn't
# survive a server restart, and grows unbounded across a long session — fine for "one
# researcher, one session," not fine as real infrastructure. See docs/PHASE2_MVP_ARCHITECTURE.md.
_RESULT_CACHE = {}


def get_cached_result(job_id):
    """Returns {"result": ..., "volume": ...} or None if not cached (e.g. server restarted
    since this job ran) — callers should fall back to the static report images in that case."""
    return _RESULT_CACHE.get(job_id)


def load_volume_from_bytes(filename: str, data: bytes, raw_dir: Path) -> np.ndarray:
    """
    Returns a (T, Z, Y, X) numpy array from either a zipped Zarr store or a multi-page TIFF
    stack. Takes raw filename + bytes rather than a Streamlit UploadedFile object specifically
    so it can be called from tests / scripts without a running Streamlit session.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / filename
    raw_path.write_bytes(data)

    if raw_path.suffix.lower() == ".zip":
        extract_dir = raw_dir / "zarr_store"
        with zipfile.ZipFile(raw_path) as zf:
            zf.extractall(extract_dir)
        import zarr
        # Assumes the zip contains a single top-level Zarr array/group. If a
        # real upload has a different layout, this is the line to adjust.
        store = zarr.open(str(extract_dir), mode="r")
        return np.asarray(store)

    elif raw_path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        volume = tifffile.imread(str(raw_path))
        # tifffile gives back whatever axis order the file declares. This
        # pipeline expects (T, Z, Y, X); a real integration needs to read
        # the TIFF's axis metadata rather than assume the shape, since
        # acquisition software varies. Flagged, not solved, here.
        if volume.ndim == 3:
            # No explicit Z axis in the file: treat every plane as a single
            # z-slice frame; adjust once real sample files are in hand.
            volume = volume[:, np.newaxis, :, :]
        return volume

    else:
        raise ValueError(
            f"Unsupported upload type: {raw_path.suffix}. "
            "Expected a .zip of a Zarr store or a .tif/.tiff stack."
        )


def run_detection_and_tracking(volume, use_cnn_filter=True, config_params=None):
    """
    The AI Processing step, minus job-status bookkeeping: detect -> (optionally) CNN-filter ->
    track -> detect divisions. Returns a LineageResult.

    config_params: optional dict overriding detection/tracking parameters (keys:
        gaussian_sigma, detection_threshold, cell_radius, tracking_max_distance,
        division_max_distance — see mvp/experiments.py's CONFIG_FIELDS). Falls back to
        src/config.py's defaults for any key not given. This is what makes a per-experiment
        Configuration (LabOS_Product_Spec_v1.md) a real feature that changes the analysis,
        rather than metadata that's recorded but never actually used.

    use_cnn_filter=False skips model loading and classify_centers entirely, running
    LineageBuilder directly on raw detector output instead — this is what lets this function
    (and everything downstream of it: tracking, visualization, report export) be smoke-tested
    without `torch` or a trained checkpoint at all. Real runs should leave this True; see
    RUNBOOK.md for why this flag exists and when to use each setting.
    """
    params = dict(config_params or {})

    detector = CellDetector(
        sigma=params.get("gaussian_sigma", config.GAUSSIAN_SIGMA),
        threshold_abs=params.get("detection_threshold", config.DETECTION_THRESHOLD),
        # int(...) is required, not cosmetic: values coming from a Configuration read out of
        # SQLite are floats (SQLite's REAL type), but skimage's peak_local_max uses min_distance
        # to build a footprint array shape and rejects a non-integer with a TypeError — found by
        # actually running a Configuration-driven experiment, not by reading the type signature.
        min_distance=int(params.get("cell_radius", config.CELL_RADIUS)),
    )
    raw_detections = detector.detect_volume(volume)

    if use_cnn_filter:
        import torch
        from src.predict import (
            load_model, collect_sample_patches, compute_sample_robust_stats,
            classify_centers_robust,
        )

        # Pass 1: build the per-frame projections once (reused for both the sample-wide patch
        # collection below and the per-frame classification pass) -- this was already being
        # recomputed per frame before (projection_t = volume[t].max(axis=0) inside the old
        # per-frame loop); computing it once up front is required now because
        # collect_sample_patches needs every frame's projection before classification can
        # start, not an incidental optimization.
        projections = [volume[t].max(axis=0) for t in range(len(raw_detections))]

        total_candidates = sum(len(c) for c in raw_detections)
        if total_candidates == 0:
            # Nothing for the CNN to do -- skip model loading and sample-stats computation
            # entirely rather than calling compute_sample_robust_stats on an empty array
            # (which raises by design; see its docstring).
            filtered_frames = raw_detections
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_model(config.BEST_MODEL_PATH, device)

            # Pass 2: collect + crop-resize every candidate across the WHOLE volume, compute
            # this sample's own median/scale ONCE from that pooled population (Run13's
            # per-sample robust normalization contract -- NOT the stored Run11 training-sample
            # statistics under results/exp05_training_dataset/run02_20samples/
            # run11_sample_robust_norm, and NOT models/norm_stats.npz's global mean/std, which
            # this path no longer reads; that file is untouched on disk and still used by
            # src/train.py and historical evaluation code).
            sample_patches = collect_sample_patches(projections, raw_detections)
            sample_median, sample_scale = compute_sample_robust_stats(sample_patches)

            # Pass 3: classify every frame's candidates using the now-known sample-wide stats.
            filtered_frames = []
            for t, coords_t in enumerate(raw_detections):
                kept_t, _ = classify_centers_robust(
                    model, projections[t], coords_t, sample_median, sample_scale, device,
                )
                filtered_frames.append(np.array(kept_t) if kept_t else np.empty((0, 2)))
    else:
        filtered_frames = raw_detections

    tracker = HungarianTracker(max_distance=params.get("tracking_max_distance", config.TRACKING_MAX_DISTANCE))
    lineage_builder = LineageBuilder(
        tracker, division_max_distance=params.get("division_max_distance", config.DIVISION_MAX_DISTANCE)
    )
    return lineage_builder.build(filtered_frames)


def process_job(job_id: str, volume: np.ndarray, use_cnn_filter=True, config_id=None):
    """
    config_id: explicit Configuration to use for THIS run, overriding the experiment's own
    config_id — this is what a v0.6.0 "Re-run Analysis" (Analysis History tab) needs: running
    the SAME experiment again with a DIFFERENT saved Configuration. When given, the
    experiment's config_id is updated to match (so Overview/Configuration tabs always show
    the config behind the latest run), while analysis_runs below preserves the true per-run
    history regardless of what "current" later becomes. None (the default) preserves the
    original behavior: use whatever config_id the experiment already has.
    """
    job_dir = STORAGE_DIR / job_id
    results_dir = job_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    status_path = job_dir / "status.json"

    def set_status(state, message=""):
        status_path.write_text(json.dumps({"state": state, "message": message}))
        # Also sync into the persistent experiment registry (mvp/experiments.py), so "My
        # Experiments" reflects real-time status without polling status.json separately —
        # a no-op if job_id isn't a registered experiment (e.g. called directly from a script).
        experiments.update_experiment(job_id, status=state)

    run_id = None
    try:
        # Look up this experiment's actual Configuration, if it has one — this is what makes
        # LabOS_Product_Spec_v1.md's Configurations table a real feature: the parameters used
        # for THIS run come from the experiment's own config_id, not just whatever
        # src/config.py's module-level defaults happen to be at the moment this runs.
        exp = experiments.get_experiment(job_id)
        effective_config_id = config_id if config_id is not None else (exp.get("config_id") if exp else None)
        config_row = experiments.get_configuration(effective_config_id) if effective_config_id else None
        config_params = (
            {k: config_row[k] for k in experiments.CONFIG_FIELDS if config_row.get(k) is not None}
            if config_row else {}
        )
        pixel_size_um = config_params.pop("pixel_size_um", None)
        frame_interval_min = config_params.pop("frame_interval_min", None)

        if config_id is not None and exp is not None:
            experiments.update_experiment(job_id, config_id=config_id)
        run_id = experiments.create_analysis_run(job_id, effective_config_id)

        experiments.log_event(job_id, "analysis_started",
                               "Detecting cells, classifying, tracking, detecting divisions...")
        set_status("processing", "Detecting cells, classifying, tracking, detecting divisions...")
        result = run_detection_and_tracking(volume, use_cnn_filter=use_cnn_filter, config_params=config_params)
        experiments.log_event(
            job_id, "tracking_finished",
            f"{len(result.nodes)} cells detected across {len(result.tracks)} track(s)",
        )

        set_status("processing", "Rendering tracked overlays, lineage tree, and report...")
        generate_report(result, results_dir, volume=volume, scores=None, sample_name=job_id,
                         pixel_size_um=pixel_size_um, frame_interval_min=frame_interval_min)
        _RESULT_CACHE[job_id] = {"result": result, "volume": volume}

        # Register this run's report as its own version (LabOS_Product_Spec_v1.md: reports
        # shouldn't be silently overwritten in place) — results_dir stays the always-current
        # "latest" copy everything else (Viewer/Analytics/Lineage) reads from; reports/{id}/
        # preserves history.
        report_id, report_dir = experiments.create_report(job_id)
        import shutil
        for filename in ("report.pdf", "tracks.csv", "ctc_tracks.csv", "summary.json"):
            src_path = results_dir / filename
            if src_path.exists():
                shutil.copy(src_path, report_dir / filename)

        experiments.log_event(job_id, "report_generated", "report.pdf generated")
        experiments.update_analysis_run(run_id, status="done", report_id=report_id)

        set_status("done", "Complete.")
        return result

    except Exception as e:
        set_status("error", str(e))
        experiments.log_event(job_id, "analysis_error", str(e))
        if run_id is not None:
            experiments.update_analysis_run(run_id, status="error", error_message=str(e))
        raise


def get_status(job_id: str) -> dict:
    status_path = STORAGE_DIR / job_id / "status.json"
    if not status_path.exists():
        return {"state": "unknown", "message": ""}
    return json.loads(status_path.read_text())
