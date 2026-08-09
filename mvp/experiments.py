"""
Persistent experiment registry — SQLite (labos.db), four tables per
LabOS_Product_Spec_v1.md's proposed schema: experiments, datasets, configurations, reports.

This is the real implementation of that spec's "Persistent Storage" and "Experiment
Management" sections, not just the design doc — datasets and configurations are genuine
first-class rows now (not a filename string and an implicit "whatever config.py said"), and
reports are versioned (each generation gets its own row + its own folder) instead of being
silently overwritten in place.

A connection is opened and closed per call rather than held open — simple and safe for SQLite
from a Streamlit app where mvp/pipeline.py's background thread and the main UI thread both
touch this module.

No `import streamlit` — this module is pure logic, testable without a running Streamlit
session, same pattern as mvp/pipeline.py.

Layout on disk:
    mvp/storage/
        labos.db                        # SQLite: experiments, datasets, configurations, reports
        {experiment_id}/
            raw/                        # uploaded file
            results/                    # LATEST results — tracks.csv, dashboard.png, frames/, etc.
            reports/{report_id}/        # one folder per generated report.pdf, versioned
            status.json                 # {"state": ..., "message": ...} (mvp/pipeline.py)
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

STORAGE_DIR = Path(__file__).resolve().parent / "storage"
DB_PATH = STORAGE_DIR / "labos.db"

# Fields configurations can hold — kept as an explicit list (rather than **kwargs everywhere)
# so a typo in a config field name fails loudly instead of silently vanishing into a JSON blob.
CONFIG_FIELDS = [
    "gaussian_sigma", "detection_threshold", "cell_radius",
    "tracking_max_distance", "division_max_distance",
    "pixel_size_um", "frame_interval_min",
]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _connect():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            researcher TEXT,
            dataset_id TEXT REFERENCES datasets(id),
            config_id TEXT REFERENCES configurations(id),
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            shape TEXT,
            uploaded_at TEXT NOT NULL,
            storage_path TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS configurations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            gaussian_sigma REAL,
            detection_threshold REAL,
            cell_radius REAL,
            tracking_max_distance REAL,
            division_max_distance REAL,
            pixel_size_um REAL,
            frame_interval_min REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            experiment_id TEXT REFERENCES experiments(id),
            generated_at TEXT NOT NULL,
            storage_path TEXT NOT NULL
        )
    """)
    # events: one row per timeline moment (experiment created, dataset uploaded, analysis
    # started/finished, report generated, ...) — v0.6.0's Timeline (per experiment) and
    # Dashboard Recent Activity (across all experiments) both read from this same table
    # rather than each inventing their own log, so "what happened" can't drift between the two.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            experiment_id TEXT REFERENCES experiments(id),
            event_type TEXT NOT NULL,
            message TEXT,
            created_at TEXT NOT NULL
        )
    """)
    # analysis_runs: v0.6.0's Analysis History — one row per time process_job() actually ran
    # for this experiment, with which Configuration it used and which report (if any) it
    # produced. Previously an experiment had exactly one implicit "run" (created alongside it,
    # config baked into experiments.config_id); this is what makes re-running the same
    # experiment with a different Configuration a real, recorded event instead of silently
    # overwriting history.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis_runs (
            id TEXT PRIMARY KEY,
            experiment_id TEXT REFERENCES experiments(id),
            config_id TEXT REFERENCES configurations(id),
            status TEXT NOT NULL,
            report_id TEXT REFERENCES reports(id),
            started_at TEXT NOT NULL,
            finished_at TEXT,
            error_message TEXT
        )
    """)
    # benchmark_runs: v0.7.0's Performance page — one row per method scored against one real
    # (usually public) dataset, e.g. a "LabOS" row and a "TrackMate" row for the same
    # Fluo-N2DL-HeLa sequence. Written by scripts/run_ctc_benchmark.py /
    # scripts/run_trackmate_comparison.py, read by mvp/streamlit_app.py::render_performance().
    # Deliberately NOT tied to any experiment_id — a benchmark run is validation evidence
    # about LabOS's pipeline in general, not a property of one researcher's one experiment.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            method TEXT NOT NULL,
            is_baseline INTEGER NOT NULL DEFAULT 0,
            adjusted_edge_jaccard REAL,
            division_jaccard REAL,
            edge_precision REAL,
            edge_recall REAL,
            node_detection_rate REAL,
            runtime_seconds REAL,
            memory_delta_mb REAL,
            notes TEXT,
            created_at TEXT NOT NULL
        )
    """)
    # Migration: is_default was added after the initial schema above, so an existing
    # labos.db from before this change won't have the column yet. ALTER TABLE ... ADD COLUMN
    # is safe to attempt every connection; only skip it if the column is already there.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(configurations)")}
    if "is_default" not in existing_cols:
        conn.execute("ALTER TABLE configurations ADD COLUMN is_default INTEGER NOT NULL DEFAULT 0")
    return conn


# --------------------------------------------------------------------------
# Events — the shared log behind both per-experiment Timeline and the
# Dashboard's cross-experiment Recent Activity feed
# --------------------------------------------------------------------------

# Event types actually logged (by create_experiment() and mvp/pipeline.py's process_job()),
# mapped to the human-readable label the UI shows. Deliberately matches only what the
# pipeline actually, distinctly does — no invented "Lineage reconstructed" or "Analytics
# exported" steps that would imply the pipeline has stages it doesn't really have.
EVENT_LABELS = {
    "experiment_created": "Experiment created",
    "dataset_uploaded": "Dataset uploaded",
    "analysis_started": "Analysis started",
    "tracking_finished": "Tracking finished",
    "report_generated": "Report generated",
    "analysis_error": "Analysis failed",
}


def log_event(experiment_id, event_type, message=None):
    event_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO events (id, experiment_id, event_type, message, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (event_id, experiment_id, event_type, message, _now()),
        )
    return event_id


def list_events(experiment_id):
    """Oldest first — a Timeline reads top-to-bottom in the order things actually happened."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, experiment_id, event_type, message, created_at FROM events "
            "WHERE experiment_id = ? ORDER BY created_at ASC",
            (experiment_id,),
        ).fetchall()
    return [
        {"id": r[0], "experiment_id": r[1], "event_type": r[2], "message": r[3], "created_at": r[4]}
        for r in rows
    ]


def list_recent_events(limit=10):
    """Most recent first, across every experiment — the Dashboard's Recent Activity feed.
    Joins in the experiment's current name (not stored on the event itself) so an activity
    line can say which experiment without a second query per row; a deleted experiment's old
    events show a placeholder name rather than erroring."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT events.id, events.experiment_id, events.event_type, events.message, "
            "events.created_at, experiments.name FROM events "
            "LEFT JOIN experiments ON events.experiment_id = experiments.id "
            "ORDER BY events.created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {"id": r[0], "experiment_id": r[1], "event_type": r[2], "message": r[3],
         "created_at": r[4], "experiment_name": r[5] or "(deleted experiment)"}
        for r in rows
    ]


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

def create_dataset(filename, shape=None, storage_path=None):
    """Registers an uploaded file as a first-class Dataset, decoupled from any one experiment
    (LabOS_Product_Spec_v1.md's reasoning: re-analyzing the same data with different
    parameters shouldn't require re-uploading it)."""
    dataset_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO datasets (id, filename, shape, uploaded_at, storage_path) "
            "VALUES (?, ?, ?, ?, ?)",
            (dataset_id, filename, json.dumps(shape) if shape else None, _now(),
             str(storage_path) if storage_path else None),
        )
    return dataset_id


def get_dataset(dataset_id):
    if dataset_id is None:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, filename, shape, uploaded_at, storage_path FROM datasets WHERE id = ?",
            (dataset_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row[0], "filename": row[1],
        "shape": json.loads(row[2]) if row[2] else None,
        "uploaded_at": row[3], "storage_path": row[4],
    }


def update_dataset(dataset_id, **fields):
    if not fields:
        return
    with _connect() as conn:
        for key, value in fields.items():
            if key == "shape":
                value = json.dumps(value)
            conn.execute(f"UPDATE datasets SET {key} = ? WHERE id = ?", (value, dataset_id))


# --------------------------------------------------------------------------
# Configurations
# --------------------------------------------------------------------------

def create_configuration(name, **params):
    """
    Creates a named, immutable snapshot of analysis parameters. Deliberately NOT deduplicated
    against existing identical configurations — every experiment gets its own configuration
    row, even if the values match a previous run's — because the point is an immutable
    historical record of "which parameters produced this specific result," which a shared/
    deduplicated row would undermine the moment anyone edited it later (see
    LabOS_Product_Spec_v1.md's Persistent Storage section for why this table exists at all).

    Unknown keys in **params raise, rather than silently vanishing — see CONFIG_FIELDS.
    """
    unknown = set(params) - set(CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"Unknown configuration field(s): {sorted(unknown)}. "
                          f"Valid fields: {CONFIG_FIELDS}")

    config_id = str(uuid.uuid4())
    values = {field: params.get(field) for field in CONFIG_FIELDS}
    with _connect() as conn:
        conn.execute(
            f"INSERT INTO configurations (id, name, created_at, {', '.join(CONFIG_FIELDS)}) "
            f"VALUES (?, ?, ?, {', '.join('?' for _ in CONFIG_FIELDS)})",
            (config_id, name, _now(), *[values[f] for f in CONFIG_FIELDS]),
        )
    return config_id


def default_configuration_params():
    """Reads the current defaults straight from src/config.py, so 'Settings' starts from what
    the system actually does today, not a guessed set of numbers."""
    import sys
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src import config

    return {
        "gaussian_sigma": config.GAUSSIAN_SIGMA,
        "detection_threshold": config.DETECTION_THRESHOLD,
        "cell_radius": config.CELL_RADIUS,
        "tracking_max_distance": config.TRACKING_MAX_DISTANCE,
        "division_max_distance": config.DIVISION_MAX_DISTANCE,
        "pixel_size_um": config.PIXEL_SIZE_UM,
        "frame_interval_min": config.FRAME_INTERVAL_MIN,
    }


def get_configuration(config_id):
    if config_id is None:
        return None
    with _connect() as conn:
        row = conn.execute(
            f"SELECT id, name, created_at, is_default, {', '.join(CONFIG_FIELDS)} "
            f"FROM configurations WHERE id = ?",
            (config_id,),
        ).fetchone()
    if row is None:
        return None
    record = {"id": row[0], "name": row[1], "created_at": row[2], "is_default": bool(row[3])}
    record.update(dict(zip(CONFIG_FIELDS, row[4:])))
    return record


def list_configurations():
    """Most recently created first — for a dropdown when starting a new experiment."""
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT id, name, created_at, is_default, {', '.join(CONFIG_FIELDS)} "
            f"FROM configurations ORDER BY created_at DESC"
        ).fetchall()
    results = []
    for row in rows:
        record = {"id": row[0], "name": row[1], "created_at": row[2], "is_default": bool(row[3])}
        record.update(dict(zip(CONFIG_FIELDS, row[4:])))
        results.append(record)
    return results


def duplicate_configuration(config_id, new_name=None):
    """Copies an existing Configuration's parameter values into a new, independent row — the
    product-review ask ("Duplicate") behind starting from a known-good config instead of
    re-typing seven numbers by hand. The copy is never the default even if the original was."""
    source = get_configuration(config_id)
    if source is None:
        raise ValueError(f"Configuration {config_id} not found")
    name = new_name or f"{source['name']} (copy)"
    return create_configuration(name, **{f: source[f] for f in CONFIG_FIELDS})


def rename_configuration(config_id, new_name):
    if not new_name or not new_name.strip():
        raise ValueError("Configuration name can't be empty")
    with _connect() as conn:
        conn.execute("UPDATE configurations SET name = ? WHERE id = ?", (new_name.strip(), config_id))


def delete_configuration(config_id):
    """Refuses to delete a Configuration that's still referenced by an experiment — an
    experiment's config_id pointing at a row that no longer exists would make
    _config_summary() silently claim 'Defaults' for a run that was never actually run with
    defaults, quietly corrupting exactly the historical record this table exists to protect
    (see create_configuration()'s docstring)."""
    with _connect() as conn:
        in_use = conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE config_id = ?", (config_id,)
        ).fetchone()[0]
        if in_use:
            raise ValueError(
                f"This Configuration is used by {in_use} experiment(s) and can't be deleted — "
                f"duplicate it if you want a variant to edit."
            )
        conn.execute("DELETE FROM configurations WHERE id = ?", (config_id,))


def set_default_configuration(config_id):
    """Marks exactly one Configuration as the default pre-selected choice in New Experiment's
    step 2 — at most one row has is_default=1 at any time."""
    with _connect() as conn:
        exists = conn.execute("SELECT 1 FROM configurations WHERE id = ?", (config_id,)).fetchone()
        if not exists:
            raise ValueError(f"Configuration {config_id} not found")
        conn.execute("UPDATE configurations SET is_default = 0")
        conn.execute("UPDATE configurations SET is_default = 1 WHERE id = ?", (config_id,))


def clear_default_configuration():
    with _connect() as conn:
        conn.execute("UPDATE configurations SET is_default = 0")


def get_default_configuration():
    with _connect() as conn:
        row = conn.execute(
            f"SELECT id, name, created_at, is_default, {', '.join(CONFIG_FIELDS)} "
            f"FROM configurations WHERE is_default = 1 LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    record = {"id": row[0], "name": row[1], "created_at": row[2], "is_default": bool(row[3])}
    record.update(dict(zip(CONFIG_FIELDS, row[4:])))
    return record


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

def create_experiment(name, researcher=None, dataset_id=None, config_id=None):
    """
    Registers a new experiment and returns its ID. Doesn't run anything — mvp/pipeline.py's
    process_job() does the actual work, keyed by this same ID, so experiments.py stays a thin
    persistence layer rather than duplicating pipeline logic.
    """
    experiment_id = str(uuid.uuid4())
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO experiments "
            "(id, name, researcher, dataset_id, config_id, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (experiment_id, name or f"Untitled experiment ({experiment_id[:8]})",
             researcher, dataset_id, config_id, "created", now, now),
        )

    experiment_dir_path = STORAGE_DIR / experiment_id
    experiment_dir_path.mkdir(parents=True, exist_ok=True)

    log_event(experiment_id, "experiment_created", f"Experiment '{name}' created" if name else None)
    if dataset_id is not None:
        dataset = get_dataset(dataset_id)
        if dataset is not None:
            log_event(experiment_id, "dataset_uploaded", dataset["filename"])

    return experiment_id


def _experiment_row_to_dict(row):
    (experiment_id, name, researcher, dataset_id, config_id, status,
     created_at, updated_at) = row
    return {
        "id": experiment_id, "name": name, "researcher": researcher,
        "dataset_id": dataset_id, "config_id": config_id, "status": status,
        "created_at": created_at, "updated_at": updated_at,
    }


def list_experiments():
    """Most-recently-created first — what 'My Experiments' actually wants to show."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, researcher, dataset_id, config_id, status, created_at, updated_at "
            "FROM experiments ORDER BY created_at DESC"
        ).fetchall()
    return [_experiment_row_to_dict(row) for row in rows]


def get_experiment(experiment_id):
    """Returns the experiment's metadata dict, or None if it doesn't exist (e.g. a stale
    link, or storage was cleared)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, name, researcher, dataset_id, config_id, status, created_at, updated_at "
            "FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()
    return _experiment_row_to_dict(row) if row else None


_EXPERIMENT_COLUMNS = {"name", "researcher", "dataset_id", "config_id", "status"}


def update_experiment(experiment_id, **fields):
    """Updates one or more of an experiment's real columns (see _EXPERIMENT_COLUMNS) and
    always bumps updated_at. Silently no-ops on an unknown experiment ID rather than raising,
    since a background thread updating status after the experiment folder was somehow removed
    shouldn't crash the whole job. Raises on an unknown FIELD name, though — that's a real bug
    in the caller, not a runtime race, and should fail loudly."""
    unknown = set(fields) - _EXPERIMENT_COLUMNS
    if unknown:
        raise ValueError(f"Unknown experiment field(s): {sorted(unknown)}. "
                          f"Valid fields: {sorted(_EXPERIMENT_COLUMNS)}")
    if not fields:
        return

    with _connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if not exists:
            return
        set_clause = ", ".join(f"{key} = ?" for key in fields)
        conn.execute(
            f"UPDATE experiments SET {set_clause}, updated_at = ? WHERE id = ?",
            (*fields.values(), _now(), experiment_id),
        )


def experiment_dir(experiment_id):
    return STORAGE_DIR / experiment_id


def results_dir(experiment_id):
    """The LATEST results — always overwritten by the most recent run, kept as the simple,
    always-current path everything (Viewer/Analytics/Lineage) reads from. Historical reports
    specifically are additionally preserved — see reports_dir()/create_report() below."""
    return experiment_dir(experiment_id) / "results"


def raw_dir(experiment_id):
    return experiment_dir(experiment_id) / "raw"


def has_results(experiment_id):
    """Whether this experiment has actually finished processing and has files on disk —
    what the Experiment Detail page's tabs check before trying to show anything."""
    rd = results_dir(experiment_id)
    return (rd / "tracks.csv").exists()


def load_lineage_result(experiment_id):
    """
    Reconstructs a LineageResult straight from the persisted tracks.csv — this is what makes
    the interactive viewer and lineage-tree highlighting keep working after a server restart,
    not just the static report.pdf. Returns None if this experiment has no results yet.
    """
    if not has_results(experiment_id):
        return None
    import pandas as pd
    import sys
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.lineage import LineageResult

    df = pd.read_csv(results_dir(experiment_id) / "tracks.csv")
    return LineageResult.from_dataframe(df)


def get_experiment_stats(experiment_id):
    """Cell/track/division counts for one experiment, for the My Experiments cards and the
    Dashboard's aggregate totals. Returns None if this experiment has no results yet (still
    running, errored, or just created) — callers should treat that as 'no numbers to show
    yet,' not zero, since zero is a real, different answer (a completed run that found
    nothing). Recomputed from the persisted tracks.csv each call rather than cached in the
    experiments table — compute_dataset_stats() is cheap (it's just counting rows/tracks, not
    re-running detection), and a cached copy would be one more place for a number to go stale."""
    if not has_results(experiment_id):
        return None
    from src.analytics import compute_dataset_stats
    result = load_lineage_result(experiment_id)
    return compute_dataset_stats(result)


# --------------------------------------------------------------------------
# Reports — versioned, not overwritten in place
# --------------------------------------------------------------------------

def create_report(experiment_id):
    """
    Reserves a new, unique folder for a report generation and registers it in the `reports`
    table — each call is a new version, never overwriting a previous one. Returns
    (report_id, report_dir): the caller (src/report.py's generate_report, via
    mvp/pipeline.py) writes the actual files into report_dir.
    """
    report_id = str(uuid.uuid4())
    report_dir_path = experiment_dir(experiment_id) / "reports" / report_id
    report_dir_path.mkdir(parents=True, exist_ok=True)

    with _connect() as conn:
        conn.execute(
            "INSERT INTO reports (id, experiment_id, generated_at, storage_path) "
            "VALUES (?, ?, ?, ?)",
            (report_id, experiment_id, _now(), str(report_dir_path)),
        )
    return report_id, report_dir_path


def list_reports(experiment_id):
    """Most recent first — the report history for one experiment."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, experiment_id, generated_at, storage_path FROM reports "
            "WHERE experiment_id = ? ORDER BY generated_at DESC",
            (experiment_id,),
        ).fetchall()
    return [
        {"id": r[0], "experiment_id": r[1], "generated_at": r[2], "storage_path": Path(r[3])}
        for r in rows
    ]


def latest_report(experiment_id):
    reports = list_reports(experiment_id)
    return reports[0] if reports else None


def list_all_reports():
    """Every report across every experiment, most recent first — Report Center (v0.6.0 Epic 3).
    Joins in the experiment's current name for display; a report whose experiment was since
    deleted shows a placeholder rather than erroring (same convention as list_recent_events)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT reports.id, reports.experiment_id, reports.generated_at, "
            "reports.storage_path, experiments.name FROM reports "
            "LEFT JOIN experiments ON reports.experiment_id = experiments.id "
            "ORDER BY reports.generated_at DESC"
        ).fetchall()
    return [
        {"id": r[0], "experiment_id": r[1], "generated_at": r[2], "storage_path": Path(r[3]),
         "experiment_name": r[4] or "(deleted experiment)"}
        for r in rows
    ]


def delete_report(report_id):
    """Removes one report version (its folder + row). Any analysis_runs row that produced this
    report keeps its own history (status/config/timestamps) — only report_id on that row goes
    stale, which list_analysis_runs' rendering needs to handle (a run can point at a
    since-deleted report), same pattern as list_recent_events' deleted-experiment placeholder."""
    import shutil
    with _connect() as conn:
        row = conn.execute("SELECT storage_path FROM reports WHERE id = ?", (report_id,)).fetchone()
        conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
    if row is not None:
        report_dir_path = Path(row[0])
        if report_dir_path.exists():
            shutil.rmtree(report_dir_path)


# --------------------------------------------------------------------------
# Analysis runs — v0.6.0's Analysis History: one row per time process_job() actually ran for
# an experiment, letting the same experiment be re-analyzed with a different Configuration
# without losing the record of what ran when, with what settings, and what it produced.
# --------------------------------------------------------------------------

_ANALYSIS_RUN_UPDATE_FIELDS = {"status", "report_id", "error_message"}


def create_analysis_run(experiment_id, config_id):
    run_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            "INSERT INTO analysis_runs "
            "(id, experiment_id, config_id, status, report_id, started_at, finished_at, error_message) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, experiment_id, config_id, "processing", None, _now(), None, None),
        )
    return run_id


def update_analysis_run(run_id, **fields):
    """Same auto-timestamp convention as update_experiment()'s updated_at: callers don't pass
    finished_at explicitly — it's set automatically the moment status becomes 'done' or
    'error', so it can't be forgotten on one call site and not another."""
    unknown = set(fields) - _ANALYSIS_RUN_UPDATE_FIELDS
    if unknown:
        raise ValueError(f"Unknown analysis_runs field(s): {unknown}")
    if not fields:
        return
    set_parts = [f"{k} = ?" for k in fields]
    values = list(fields.values())
    if fields.get("status") in ("done", "error"):
        set_parts.append("finished_at = ?")
        values.append(_now())
    values.append(run_id)
    with _connect() as conn:
        conn.execute(f"UPDATE analysis_runs SET {', '.join(set_parts)} WHERE id = ?", values)


def list_analysis_runs(experiment_id):
    """Most recent first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, experiment_id, config_id, status, report_id, started_at, finished_at, "
            "error_message FROM analysis_runs WHERE experiment_id = ? ORDER BY started_at DESC",
            (experiment_id,),
        ).fetchall()
    return [
        {"id": r[0], "experiment_id": r[1], "config_id": r[2], "status": r[3], "report_id": r[4],
         "started_at": r[5], "finished_at": r[6], "error_message": r[7]}
        for r in rows
    ]


# --------------------------------------------------------------------------
# Benchmark runs — v0.7.0's Performance page: real validation evidence (LabOS scored against
# a real dataset's ground truth, alongside a baseline like TrackMate scored the same way),
# not per-experiment history like analysis_runs above.
# --------------------------------------------------------------------------

_BENCHMARK_METRIC_FIELDS = [
    "adjusted_edge_jaccard", "division_jaccard", "edge_precision", "edge_recall",
    "node_detection_rate", "runtime_seconds", "memory_delta_mb",
]


def record_benchmark_run(dataset, method, is_baseline=False, notes=None, **metrics):
    """metrics: any of _BENCHMARK_METRIC_FIELDS, all optional (e.g. a baseline run's
    memory_delta_mb might genuinely be unmeasured, same reasoning as
    src/benchmark.py::score_external_method's runtime_seconds=None case — an unmeasured metric
    is recorded as missing, never fabricated as 0)."""
    unknown = set(metrics) - set(_BENCHMARK_METRIC_FIELDS)
    if unknown:
        raise ValueError(f"Unknown benchmark metric(s): {unknown}")
    run_id = str(uuid.uuid4())
    values = [metrics.get(f) for f in _BENCHMARK_METRIC_FIELDS]
    with _connect() as conn:
        conn.execute(
            f"INSERT INTO benchmark_runs "
            f"(id, dataset, method, is_baseline, {', '.join(_BENCHMARK_METRIC_FIELDS)}, notes, created_at) "
            f"VALUES (?, ?, ?, ?, {', '.join(['?'] * len(_BENCHMARK_METRIC_FIELDS))}, ?, ?)",
            [run_id, dataset, method, int(bool(is_baseline))] + values + [notes, _now()],
        )
    return run_id


def list_benchmark_runs():
    """Most recent first — the Performance page groups these by dataset for display."""
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT id, dataset, method, is_baseline, {', '.join(_BENCHMARK_METRIC_FIELDS)}, "
            f"notes, created_at FROM benchmark_runs ORDER BY created_at DESC"
        ).fetchall()
    results = []
    for r in rows:
        record = {"id": r[0], "dataset": r[1], "method": r[2], "is_baseline": bool(r[3])}
        record.update(dict(zip(_BENCHMARK_METRIC_FIELDS, r[4:4 + len(_BENCHMARK_METRIC_FIELDS)])))
        record["notes"] = r[4 + len(_BENCHMARK_METRIC_FIELDS)]
        record["created_at"] = r[5 + len(_BENCHMARK_METRIC_FIELDS)]
        results.append(record)
    return results


def delete_benchmark_run(run_id):
    with _connect() as conn:
        conn.execute("DELETE FROM benchmark_runs WHERE id = ?", (run_id,))


# --------------------------------------------------------------------------
# Deletion
# --------------------------------------------------------------------------

def delete_experiment(experiment_id):
    """Removes an experiment's files and its database rows (experiments + reports + events +
    analysis_runs; datasets and configurations are left alone since they may be referenced by
    other experiments). Not exposed as a one-click button anywhere yet — deliberately no UI
    for this in v0, since accidental data loss is worse than a slightly cluttered list; add
    the button once someone actually asks for it."""
    import shutil
    with _connect() as conn:
        conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        conn.execute("DELETE FROM reports WHERE experiment_id = ?", (experiment_id,))
        conn.execute("DELETE FROM events WHERE experiment_id = ?", (experiment_id,))
        conn.execute("DELETE FROM analysis_runs WHERE experiment_id = ?", (experiment_id,))
    d = experiment_dir(experiment_id)
    if d.exists():
        shutil.rmtree(d)
