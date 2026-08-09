import sys
import importlib
from pathlib import Path

import numpy as np
import tifffile


def _fresh_experiments_module(tmp_path):
    """Reloads mvp.experiments pointed at a throwaway storage dir, so tests don't touch the
    real mvp/storage/ and don't leak state between tests."""
    if "mvp.experiments" in sys.modules:
        del sys.modules["mvp.experiments"]
    import mvp.experiments as experiments
    experiments.STORAGE_DIR = tmp_path / "storage"
    experiments.DB_PATH = experiments.STORAGE_DIR / "labos.db"
    return experiments


def test_create_and_get_experiment(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Embryo A")
    exp = experiments.get_experiment(exp_id)
    assert exp["name"] == "Embryo A"
    assert exp["status"] == "created"
    assert exp["id"] == exp_id


def test_untitled_experiment_gets_a_default_name(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment(None)
    exp = experiments.get_experiment(exp_id)
    assert "Untitled" in exp["name"]


def test_list_experiments_most_recent_first(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    id_a = experiments.create_experiment("First")
    id_b = experiments.create_experiment("Second")
    listing = experiments.list_experiments()
    assert listing[0]["id"] == id_b
    assert listing[1]["id"] == id_a


def test_update_experiment_persists(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Embryo A")
    experiments.update_experiment(exp_id, status="processing")
    assert experiments.get_experiment(exp_id)["status"] == "processing"


def test_update_unknown_experiment_does_not_raise(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    experiments.update_experiment("does-not-exist", status="done")  # should not raise


def test_persistence_survives_a_simulated_restart(tmp_path):
    """The actual point of this whole module: reload it fresh, as a new Python process
    would, and confirm the data is still there — not just held in memory."""
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Drug Study 01")
    experiments.update_experiment(exp_id, status="done")

    del sys.modules["mvp.experiments"]
    import mvp.experiments as experiments_reloaded
    experiments_reloaded.STORAGE_DIR = tmp_path / "storage"
    experiments_reloaded.DB_PATH = experiments_reloaded.STORAGE_DIR / "labos.db"

    reloaded = experiments_reloaded.get_experiment(exp_id)
    assert reloaded is not None
    assert reloaded["name"] == "Drug Study 01"
    assert reloaded["status"] == "done"


def test_has_results_false_before_processing(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Embryo A")
    assert experiments.has_results(exp_id) is False


def test_load_lineage_result_returns_none_without_results(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Embryo A")
    assert experiments.load_lineage_result(exp_id) is None


def test_full_pipeline_to_persisted_result_round_trip(tmp_path):
    """The real chain: create an experiment, run the actual pipeline, persist a report,
    simulate a restart, and confirm the interactive-viewer-ready LineageResult reloads
    correctly — including a real division, not a trivial case."""
    experiments = _fresh_experiments_module(tmp_path)

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from mvp.pipeline import run_detection_and_tracking
    from src.report import generate_report

    exp_id = experiments.create_experiment("Round Trip Test")
    volume = tifffile.imread(str(project_root / "mvp" / "test_data" / "synthetic_test.tif"))
    if volume.ndim == 3:
        volume = volume[:, None, :, :]

    result = run_detection_and_tracking(volume, use_cnn_filter=False)
    generate_report(result, experiments.results_dir(exp_id), volume=volume, sample_name=exp_id)
    experiments.update_experiment(exp_id, status="done")

    assert experiments.has_results(exp_id)

    del sys.modules["mvp.experiments"]
    import mvp.experiments as experiments_reloaded
    experiments_reloaded.STORAGE_DIR = tmp_path / "storage"
    experiments_reloaded.DB_PATH = experiments_reloaded.STORAGE_DIR / "labos.db"

    reloaded_result = experiments_reloaded.load_lineage_result(exp_id)
    assert reloaded_result is not None
    assert len(reloaded_result.nodes) == len(result.nodes)
    assert reloaded_result.n_divisions() == result.n_divisions()
    assert reloaded_result.n_divisions() >= 1  # this specific file has a real division

    # descendant highlighting must also work post-reload, not just raw counts
    root = min(t for t, info in reloaded_result.tracks.items() if info["parent_track_id"] is None)
    assert len(reloaded_result.get_descendant_track_ids(root)) >= 1


def test_delete_experiment_removes_it(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Temporary")
    assert experiments.get_experiment(exp_id) is not None
    experiments.delete_experiment(exp_id)
    assert experiments.get_experiment(exp_id) is None
    assert not experiments.experiment_dir(exp_id).exists()


def test_storage_is_a_real_sqlite_file_named_labos_db(tmp_path):
    """Confirms this is actually SQLite on disk (labos.db), not just an API that happens to
    look the same — opens the file independently with the stdlib sqlite3 module and queries
    it directly, bypassing mvp.experiments entirely."""
    import sqlite3

    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Real DB Check")

    db_path = tmp_path / "storage" / "labos.db"
    assert db_path.exists()
    assert db_path.read_bytes()[:16] == b"SQLite format 3\x00"  # the actual SQLite file magic bytes

    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT id FROM experiments WHERE id = ?", (exp_id,)).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == exp_id


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

def test_create_and_get_dataset(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    ds_id = experiments.create_dataset("embryo_a.tif", shape=[8, 3, 96, 96], storage_path="raw/embryo_a.tif")
    ds = experiments.get_dataset(ds_id)
    assert ds["filename"] == "embryo_a.tif"
    assert ds["shape"] == [8, 3, 96, 96]
    assert ds["storage_path"] == "raw/embryo_a.tif"


def test_dataset_requires_storage_path(tmp_path):
    """storage_path is NOT NULL in the schema on purpose — a dataset with no known file
    location isn't a usable dataset. Caught by an actual SQLite IntegrityError while building
    this, not just designed in the abstract."""
    experiments = _fresh_experiments_module(tmp_path)
    import sqlite3
    try:
        experiments.create_dataset("no_path.tif", shape=[1, 1, 1, 1])
        assert False, "expected an IntegrityError for a missing storage_path"
    except sqlite3.IntegrityError:
        pass


def test_dataset_survives_a_simulated_restart(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    ds_id = experiments.create_dataset("embryo_a.tif", shape=[8, 3, 96, 96], storage_path="raw/embryo_a.tif")

    del sys.modules["mvp.experiments"]
    import mvp.experiments as experiments_reloaded
    experiments_reloaded.STORAGE_DIR = tmp_path / "storage"
    experiments_reloaded.DB_PATH = experiments_reloaded.STORAGE_DIR / "labos.db"

    ds = experiments_reloaded.get_dataset(ds_id)
    assert ds is not None
    assert ds["filename"] == "embryo_a.tif"


# --------------------------------------------------------------------------
# Configurations
# --------------------------------------------------------------------------

def test_create_configuration_from_real_config_py_defaults(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    defaults = experiments.default_configuration_params()
    config_id = experiments.create_configuration("Default", **defaults)
    cfg = experiments.get_configuration(config_id)
    assert cfg["gaussian_sigma"] == defaults["gaussian_sigma"]
    assert cfg["detection_threshold"] == defaults["detection_threshold"]
    assert cfg["name"] == "Default"


def test_create_configuration_rejects_unknown_field(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    try:
        experiments.create_configuration("Bad", not_a_real_field=1)
        assert False, "expected a ValueError for an unknown configuration field"
    except ValueError as e:
        assert "not_a_real_field" in str(e)


def test_configurations_are_not_deduplicated(tmp_path):
    """Deliberate design choice (see create_configuration's docstring): every experiment gets
    its own immutable configuration row, even with identical values, so a later edit to one
    experiment's parameters can never retroactively change what an earlier experiment says it
    used."""
    experiments = _fresh_experiments_module(tmp_path)
    defaults = experiments.default_configuration_params()
    id_1 = experiments.create_configuration("Run 1", **defaults)
    id_2 = experiments.create_configuration("Run 2", **defaults)
    assert id_1 != id_2


def test_list_configurations_most_recent_first(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    defaults = experiments.default_configuration_params()
    experiments.create_configuration("First", **defaults)
    experiments.create_configuration("Second", **defaults)
    listing = experiments.list_configurations()
    assert listing[0]["name"] == "Second"
    assert listing[1]["name"] == "First"


def test_configuration_survives_a_simulated_restart(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    defaults = experiments.default_configuration_params()
    config_id = experiments.create_configuration("Persisted Config", **defaults)

    del sys.modules["mvp.experiments"]
    import mvp.experiments as experiments_reloaded
    experiments_reloaded.STORAGE_DIR = tmp_path / "storage"
    experiments_reloaded.DB_PATH = experiments_reloaded.STORAGE_DIR / "labos.db"

    cfg = experiments_reloaded.get_configuration(config_id)
    assert cfg is not None
    assert cfg["name"] == "Persisted Config"


# --------------------------------------------------------------------------
# Experiments: researcher, dataset_id, config_id, updated_at
# --------------------------------------------------------------------------

def test_experiment_can_reference_a_researcher_dataset_and_configuration(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    ds_id = experiments.create_dataset("embryo_a.tif", shape=[8, 3, 96, 96], storage_path="raw/embryo_a.tif")
    defaults = experiments.default_configuration_params()
    config_id = experiments.create_configuration("Default", **defaults)

    exp_id = experiments.create_experiment("Embryo Development Day 3", researcher="Dr. Ada Lovelace",
                                            dataset_id=ds_id, config_id=config_id)
    exp = experiments.get_experiment(exp_id)
    assert exp["researcher"] == "Dr. Ada Lovelace"
    assert exp["dataset_id"] == ds_id
    assert exp["config_id"] == config_id


def test_experiment_created_at_equals_updated_at_initially(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Embryo A")
    exp = experiments.get_experiment(exp_id)
    assert exp["created_at"] == exp["updated_at"]


def test_update_experiment_bumps_updated_at_but_not_created_at(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Embryo A")
    original = experiments.get_experiment(exp_id)

    import time
    time.sleep(0.01)
    experiments.update_experiment(exp_id, status="processing")

    updated = experiments.get_experiment(exp_id)
    assert updated["created_at"] == original["created_at"]
    assert updated["updated_at"] != original["updated_at"]


def test_update_experiment_rejects_unknown_field(tmp_path):
    """A typo in a field name should fail loudly, not silently vanish — this is a real
    behavior change from the earlier JSON-blob-based schema, where any field name was
    silently accepted."""
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Embryo A")
    try:
        experiments.update_experiment(exp_id, not_a_real_column="x")
        assert False, "expected a ValueError for an unknown experiment field"
    except ValueError as e:
        assert "not_a_real_column" in str(e)


# --------------------------------------------------------------------------
# Reports — versioned, not overwritten
# --------------------------------------------------------------------------

def test_create_report_returns_a_unique_id_and_folder_each_time(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Embryo A")

    report_id_1, report_dir_1 = experiments.create_report(exp_id)
    report_id_2, report_dir_2 = experiments.create_report(exp_id)

    assert report_id_1 != report_id_2
    assert report_dir_1 != report_dir_2
    assert report_dir_1.exists()
    assert report_dir_2.exists()


def test_reports_are_not_overwritten_in_place(tmp_path):
    """The actual point: writing a second report must not touch the first one's files."""
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Embryo A")

    _, report_dir_1 = experiments.create_report(exp_id)
    (report_dir_1 / "report.pdf").write_text("version 1")
    _, report_dir_2 = experiments.create_report(exp_id)
    (report_dir_2 / "report.pdf").write_text("version 2")

    assert (report_dir_1 / "report.pdf").read_text() == "version 1"
    assert (report_dir_2 / "report.pdf").read_text() == "version 2"


def test_list_reports_most_recent_first(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Embryo A")
    id_1, _ = experiments.create_report(exp_id)
    id_2, _ = experiments.create_report(exp_id)

    reports = experiments.list_reports(exp_id)
    assert len(reports) == 2
    assert reports[0]["id"] == id_2
    assert reports[1]["id"] == id_1


def test_latest_report_returns_the_most_recent(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Embryo A")
    experiments.create_report(exp_id)
    id_2, _ = experiments.create_report(exp_id)
    assert experiments.latest_report(exp_id)["id"] == id_2


def test_latest_report_none_when_no_reports_exist(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Embryo A")
    assert experiments.latest_report(exp_id) is None


# --------------------------------------------------------------------------
# The real, end-to-end proof: a Configuration actually changes detection results
# --------------------------------------------------------------------------

def test_configuration_actually_changes_detection_results(tmp_path):
    """Not cosmetic metadata: two experiments on the same data with different Configurations
    (a normal detection threshold vs. an absurdly strict one) must produce different results.
    This is the test that caught a real int-vs-float bug (SQLite returns REAL columns as
    Python floats, but skimage's peak_local_max requires an int for min_distance) while
    building this feature."""
    experiments = _fresh_experiments_module(tmp_path)
    import mvp.pipeline as pipeline
    pipeline.STORAGE_DIR = experiments.STORAGE_DIR
    pipeline.experiments.STORAGE_DIR = experiments.STORAGE_DIR
    pipeline.experiments.DB_PATH = experiments.DB_PATH

    project_root = Path(__file__).resolve().parent.parent
    tiff_path = project_root / "mvp" / "test_data" / "synthetic_test.tif"
    dataset_id = experiments.create_dataset("synthetic_test.tif", shape=[6, 3, 96, 96],
                                              storage_path=str(tiff_path))

    defaults = experiments.default_configuration_params()
    config_normal = experiments.create_configuration("Normal", **defaults)
    strict_params = dict(defaults)
    strict_params["detection_threshold"] = 50000  # absurdly high -- should suppress everything
    config_strict = experiments.create_configuration("Very Strict", **strict_params)

    volume = tifffile.imread(str(tiff_path))
    if volume.ndim == 3:
        volume = volume[:, None, :, :]

    exp_normal = experiments.create_experiment("Normal run", dataset_id=dataset_id, config_id=config_normal)
    pipeline.process_job(exp_normal, volume, use_cnn_filter=False)
    result_normal = experiments.load_lineage_result(exp_normal)

    exp_strict = experiments.create_experiment("Strict run", dataset_id=dataset_id, config_id=config_strict)
    pipeline.process_job(exp_strict, volume, use_cnn_filter=False)
    result_strict = experiments.load_lineage_result(exp_strict)

    assert len(result_strict.nodes) < len(result_normal.nodes)
    assert len(result_strict.nodes) == 0  # this specific threshold suppresses everything
    assert experiments.get_experiment(exp_strict)["status"] == "done"  # didn't crash on 0 detections


def test_process_job_creates_a_new_report_version_each_run(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    import mvp.pipeline as pipeline
    pipeline.STORAGE_DIR = experiments.STORAGE_DIR
    pipeline.experiments.STORAGE_DIR = experiments.STORAGE_DIR
    pipeline.experiments.DB_PATH = experiments.DB_PATH

    project_root = Path(__file__).resolve().parent.parent
    tiff_path = project_root / "mvp" / "test_data" / "synthetic_test.tif"
    volume = tifffile.imread(str(tiff_path))
    if volume.ndim == 3:
        volume = volume[:, None, :, :]

    exp_id = experiments.create_experiment("Re-run test")
    pipeline.process_job(exp_id, volume, use_cnn_filter=False)
    assert len(experiments.list_reports(exp_id)) == 1

    pipeline.process_job(exp_id, volume, use_cnn_filter=False)
    assert len(experiments.list_reports(exp_id)) == 2


# --------------------------------------------------------------------------
# Configuration management: Duplicate / Rename / Delete / Set as Default
# (product-review request: mvp/experiments.py's config rows were previously write-once)
# --------------------------------------------------------------------------

def test_duplicate_configuration_copies_params_into_a_new_row(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    defaults = experiments.default_configuration_params()
    original_id = experiments.create_configuration("Original", **defaults)

    copy_id = experiments.duplicate_configuration(original_id)
    assert copy_id != original_id
    copy = experiments.get_configuration(copy_id)
    assert copy["name"] == "Original (copy)"
    assert copy["gaussian_sigma"] == defaults["gaussian_sigma"]
    assert copy["is_default"] is False  # never inherits default status


def test_duplicate_configuration_accepts_a_custom_name(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    defaults = experiments.default_configuration_params()
    original_id = experiments.create_configuration("Original", **defaults)
    copy_id = experiments.duplicate_configuration(original_id, new_name="High Sensitivity v2")
    assert experiments.get_configuration(copy_id)["name"] == "High Sensitivity v2"


def test_rename_configuration(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    defaults = experiments.default_configuration_params()
    config_id = experiments.create_configuration("Old Name", **defaults)
    experiments.rename_configuration(config_id, "New Name")
    assert experiments.get_configuration(config_id)["name"] == "New Name"


def test_rename_configuration_rejects_empty_name(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    defaults = experiments.default_configuration_params()
    config_id = experiments.create_configuration("Keep Me", **defaults)
    try:
        experiments.rename_configuration(config_id, "   ")
        assert False, "expected a ValueError for a blank name"
    except ValueError:
        pass
    assert experiments.get_configuration(config_id)["name"] == "Keep Me"


def test_delete_configuration_removes_an_unused_row(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    defaults = experiments.default_configuration_params()
    config_id = experiments.create_configuration("Unused", **defaults)
    experiments.delete_configuration(config_id)
    assert experiments.get_configuration(config_id) is None


def test_delete_configuration_refuses_if_an_experiment_uses_it(tmp_path):
    """The whole point of Configuration rows being immutable history (see
    create_configuration's docstring) breaks if one can vanish out from under an experiment
    that still points at it."""
    experiments = _fresh_experiments_module(tmp_path)
    defaults = experiments.default_configuration_params()
    config_id = experiments.create_configuration("In Use", **defaults)
    experiments.create_experiment("Uses it", config_id=config_id)

    try:
        experiments.delete_configuration(config_id)
        assert False, "expected a ValueError when the configuration is still referenced"
    except ValueError as e:
        assert "experiment" in str(e).lower()
    assert experiments.get_configuration(config_id) is not None  # still there


def test_set_default_configuration_is_exclusive(tmp_path):
    """At most one Configuration is ever the default at a time — setting a second one clears
    the first, rather than both ending up marked default."""
    experiments = _fresh_experiments_module(tmp_path)
    defaults = experiments.default_configuration_params()
    id_a = experiments.create_configuration("A", **defaults)
    id_b = experiments.create_configuration("B", **defaults)

    experiments.set_default_configuration(id_a)
    assert experiments.get_configuration(id_a)["is_default"] is True
    assert experiments.get_default_configuration()["id"] == id_a

    experiments.set_default_configuration(id_b)
    assert experiments.get_configuration(id_a)["is_default"] is False
    assert experiments.get_configuration(id_b)["is_default"] is True
    assert experiments.get_default_configuration()["id"] == id_b


def test_clear_default_configuration(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    defaults = experiments.default_configuration_params()
    config_id = experiments.create_configuration("A", **defaults)
    experiments.set_default_configuration(config_id)
    assert experiments.get_default_configuration() is not None
    experiments.clear_default_configuration()
    assert experiments.get_default_configuration() is None


def test_configurations_table_migrates_in_is_default_column(tmp_path):
    """A labos.db created before this feature existed won't have the is_default column yet —
    _connect() must add it via ALTER TABLE rather than erroring on every call for anyone
    upgrading LabOS with an existing database."""
    experiments = _fresh_experiments_module(tmp_path)
    import sqlite3
    experiments.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(experiments.DB_PATH))
    conn.execute("""
        CREATE TABLE configurations (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL,
            gaussian_sigma REAL, detection_threshold REAL, cell_radius REAL,
            tracking_max_distance REAL, division_max_distance REAL,
            pixel_size_um REAL, frame_interval_min REAL
        )
    """)
    conn.execute(
        "INSERT INTO configurations VALUES ('abc','Pre-migration','2026-01-01T00:00:00', "
        "2.0, 800, 12, 25, 37.5, 0, 0)"
    )
    conn.commit()
    conn.close()

    cfg = experiments.get_configuration("abc")
    assert cfg is not None
    assert cfg["is_default"] is False
    experiments.set_default_configuration("abc")
    assert experiments.get_configuration("abc")["is_default"] is True


def test_get_experiment_stats_returns_none_before_results_exist(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("No results yet")
    assert experiments.get_experiment_stats(exp_id) is None


# --------------------------------------------------------------------------
# v0.6.0: events (Timeline / Recent Activity) and analysis_runs (Analysis History)
# --------------------------------------------------------------------------

def test_create_experiment_logs_creation_and_dataset_events(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    ds_id = experiments.create_dataset("embryo.tif", shape=[3, 1, 32, 32], storage_path="raw/embryo.tif")
    exp_id = experiments.create_experiment("Embryo Day 3", dataset_id=ds_id)

    events = experiments.list_events(exp_id)
    assert [e["event_type"] for e in events] == ["experiment_created", "dataset_uploaded"]
    assert events[0]["created_at"] <= events[1]["created_at"]  # oldest first


def test_create_experiment_without_dataset_only_logs_creation(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("No dataset yet")
    events = experiments.list_events(exp_id)
    assert [e["event_type"] for e in events] == ["experiment_created"]


def test_log_event_and_list_events_are_chronological(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Exp")
    experiments.log_event(exp_id, "analysis_started", "msg 1")
    experiments.log_event(exp_id, "tracking_finished", "msg 2")
    events = experiments.list_events(exp_id)
    assert [e["event_type"] for e in events] == ["experiment_created", "analysis_started", "tracking_finished"]
    assert events[1]["message"] == "msg 1"


def test_list_recent_events_is_most_recent_first_across_experiments(tmp_path):
    """The Dashboard's Recent Activity feed spans every experiment, not just one — and needs
    each experiment's current name without a second query per row."""
    experiments = _fresh_experiments_module(tmp_path)
    exp_a = experiments.create_experiment("Exp A")
    exp_b = experiments.create_experiment("Exp B")
    experiments.log_event(exp_a, "analysis_started")

    recent = experiments.list_recent_events(limit=10)
    assert recent[0]["event_type"] == "analysis_started"
    assert recent[0]["experiment_name"] == "Exp A"
    assert recent[-1]["experiment_name"] in ("Exp A", "Exp B")  # both experiments' creation events present


def test_list_recent_events_respects_limit(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Exp")
    for i in range(5):
        experiments.log_event(exp_id, "analysis_started", f"run {i}")
    assert len(experiments.list_recent_events(limit=3)) == 3


def test_create_and_update_analysis_run(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    defaults = experiments.default_configuration_params()
    config_id = experiments.create_configuration("Cfg", **defaults)
    exp_id = experiments.create_experiment("Exp", config_id=config_id)

    run_id = experiments.create_analysis_run(exp_id, config_id)
    runs = experiments.list_analysis_runs(exp_id)
    assert len(runs) == 1
    assert runs[0]["status"] == "processing"
    assert runs[0]["finished_at"] is None

    experiments.update_analysis_run(run_id, status="done", report_id="report-123")
    runs = experiments.list_analysis_runs(exp_id)
    assert runs[0]["status"] == "done"
    assert runs[0]["report_id"] == "report-123"
    assert runs[0]["finished_at"] is not None  # auto-set, not passed explicitly


def test_update_analysis_run_error_status_sets_finished_at_and_message(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Exp")
    run_id = experiments.create_analysis_run(exp_id, None)
    experiments.update_analysis_run(run_id, status="error", error_message="boom")
    run = experiments.list_analysis_runs(exp_id)[0]
    assert run["status"] == "error"
    assert run["error_message"] == "boom"
    assert run["finished_at"] is not None


def test_update_analysis_run_rejects_unknown_field(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Exp")
    run_id = experiments.create_analysis_run(exp_id, None)
    try:
        experiments.update_analysis_run(run_id, config_id="not-allowed-to-change-here")
        assert False, "expected a ValueError for an unrecognized field"
    except ValueError:
        pass


def test_list_analysis_runs_most_recent_first(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Exp")
    run_a = experiments.create_analysis_run(exp_id, None)
    run_b = experiments.create_analysis_run(exp_id, None)
    runs = experiments.list_analysis_runs(exp_id)
    assert runs[0]["id"] == run_b
    assert runs[1]["id"] == run_a


def test_list_all_reports_spans_every_experiment(tmp_path):
    """Report Center (v0.6.0 Epic 3): every report, from every experiment, in one place."""
    experiments = _fresh_experiments_module(tmp_path)
    exp_a = experiments.create_experiment("Exp A")
    exp_b = experiments.create_experiment("Exp B")
    experiments.create_report(exp_a)
    experiments.create_report(exp_b)

    all_reports = experiments.list_all_reports()
    assert len(all_reports) == 2
    names = {r["experiment_name"] for r in all_reports}
    assert names == {"Exp A", "Exp B"}


def test_delete_report_removes_row_and_folder(tmp_path):
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Exp")
    report_id, report_dir = experiments.create_report(exp_id)
    (report_dir / "report.pdf").write_bytes(b"%PDF-1.4 fake")
    assert report_dir.exists()

    experiments.delete_report(report_id)
    assert experiments.list_reports(exp_id) == []
    assert not report_dir.exists()


def test_delete_experiment_also_clears_events_and_analysis_runs(tmp_path):
    """delete_experiment()'s docstring says it clears events + analysis_runs too — this is the
    test for that claim, not just the reports/experiments rows it always cleared."""
    experiments = _fresh_experiments_module(tmp_path)
    exp_id = experiments.create_experiment("Exp")
    experiments.create_analysis_run(exp_id, None)
    experiments.log_event(exp_id, "analysis_started")

    experiments.delete_experiment(exp_id)
    assert experiments.list_events(exp_id) == []
    assert experiments.list_analysis_runs(exp_id) == []
