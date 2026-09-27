"""LabOS — Research Workspace (v0).

An AI-powered workspace for live-cell imaging research, built on the LabOS
cell-tracking pipeline in ``src/``.

Dashboard → My Experiments → New Experiment → Experiment Detail, backed by
the persistent SQLite registry in ``mvp/experiments.py``.

The raw uploaded volume is held in memory for the active session and is not
persisted across a server restart. Persisted results include lineage,
analytics, reports/exports, analysis history, timeline events, and rendered
static result frames. The interactive viewer and Re-run Analysis require
the raw volume to still be available in the current session.

Run with:
    streamlit run mvp/streamlit_app.py
"""

import json
import threading
import time
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

import mvp.experiments as experiments
from mvp.pipeline import STORAGE_DIR, load_volume_from_bytes, process_job, get_status, get_cached_result
from mvp.formatting import format_relative_time

STORAGE_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="LabOS", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "landing"
if "current_experiment_id" not in st.session_state:
    st.session_state.current_experiment_id = None
if "pending_dataset" not in st.session_state:
    st.session_state.pending_dataset = None  # {dataset_id}, set once step 2 (Dataset) is done
if "experiment_info" not in st.session_state:
    st.session_state.experiment_info = None  # {name, researcher}, set once step 1 is done
if "_volume_cache" not in st.session_state:
    # Raw volumes for datasets awaiting a Run Analysis decision, keyed by dataset_id — small,
    # in-memory, cleared (via .pop()) once the experiment is created. Must be session_state,
    # not a plain module-level dict: Streamlit reruns this whole script top-to-bottom on every
    # interaction, so a plain `_VOLUME_CACHE = {}` at module level is reset on every single
    # rerun — including the rerun between "cache the volume" (step 2) and "read it back"
    # (step 3's Run Analysis button), which silently turned every analysis into
    # process_job(experiment_id, None). Not part of mvp/experiments.py's persistence story on
    # purpose: this is transient wizard state, not a first-class persisted object.
    st.session_state._volume_cache = {}


def _go(page, experiment_id=None):
    st.session_state.page = page
    if experiment_id is not None:
        st.session_state.current_experiment_id = experiment_id
    st.rerun()


def _reset_new_experiment_wizard():
    """Clears both New Experiment steps' state — called anywhere 'New Experiment' is started
    fresh, so a half-finished previous attempt (e.g. name typed, then navigated away) doesn't
    leak into the next one."""
    st.session_state.pending_dataset = None
    st.session_state.experiment_info = None


def _render_wizard_progress(current_step):
    """Step 1 Experiment Info -> Step 2 Dataset -> Step 3 Configuration, per the product
    review's request for a visible stepper instead of one long form."""
    labels = ["1. Experiment Info", "2. Dataset", "3. Configuration"]
    cols = st.columns(3)
    for i, (col, label) in enumerate(zip(cols, labels), start=1):
        with col:
            if i < current_step:
                st.markdown(f"✅ ~~{label}~~")
            elif i == current_step:
                st.markdown(f"**➡️ {label}**")
            else:
                st.markdown(f":gray[{label}]")
    st.divider()


def _status_badge(status):
    colors = {"created": "⚪", "queued": "⚪", "processing": "🟡", "done": "🟢", "error": "🔴"}
    return f"{colors.get(status, '⚪')} {status.capitalize()}"


def _config_summary(cfg):
    if cfg is None:
        return "Defaults (no saved Configuration)"
    parts = [f"σ={cfg['gaussian_sigma']}", f"threshold={cfg['detection_threshold']}",
              f"radius={cfg['cell_radius']}"]
    if cfg.get("pixel_size_um") and cfg.get("frame_interval_min"):
        parts.append(f"{cfg['pixel_size_um']} µm/px, {cfg['frame_interval_min']} min/frame")
    return f"{cfg['name']} ({', '.join(parts)})"


# ============================================================================
# Sidebar — persistent across every page
# ============================================================================
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.png"


def render_sidebar():
    with st.sidebar:
        if LOGO_PATH.exists():
            col_logo, col_name = st.columns([1, 3])
            with col_logo:
                st.image(str(LOGO_PATH), width=40)
            with col_name:
                st.markdown("## LabOS")
        else:
            st.markdown("## 🧬 LabOS")
        st.caption("Research OS")
        st.divider()
        if st.button("🏠 Dashboard", use_container_width=True):
            _go("landing")
        if st.button("🧪 My Experiments", use_container_width=True):
            _go("list")
        if st.button("+ New Experiment", use_container_width=True, type="primary"):
            _reset_new_experiment_wizard()
            _go("new")
        if st.button("📄 Reports", use_container_width=True):
            _go("reports")
        if st.button("📊 Performance", use_container_width=True):
            _go("performance")
        if st.button("⚙️ Settings", use_container_width=True):
            _go("settings")


# ============================================================================
# Landing page (Dashboard)
# ============================================================================
def render_landing():
    st.markdown(
        "<h1 style='margin-bottom:0'>LabOS — The Research Operating System</h1>"
        "<p style='font-size:1.15em;color:gray;margin-top:0'>"
        "Manage microscopy experiments, analyze cell dynamics, reconstruct lineages, and "
        "generate publication-ready reports from one unified workspace.</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    all_experiments = experiments.list_experiments()
    done = [e for e in all_experiments if e.get("status") == "done"]
    processing = [e for e in all_experiments if e.get("status") in ("queued", "processing")]

    # Real aggregates, not just row counts — computed from each completed experiment's own
    # persisted results (mvp/experiments.get_experiment_stats), same numbers each Experiment
    # Detail's Overview tab shows, just summed across every experiment. An experiment still
    # queued/processing/errored contributes nothing here (get_experiment_stats returns None
    # for it) rather than a misleading zero.
    total_cells = 0
    total_divisions = 0
    for e in done:
        stats = experiments.get_experiment_stats(e["id"])
        if stats:
            total_cells += stats["n_cells_detected"]
            total_divisions += stats["n_divisions"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Experiments", len(all_experiments),
                help=f"{len(done)} completed, {len(processing)} running" if all_experiments else None)
    col2.metric("🧬 Cells Tracked", f"{total_cells:,}")
    col3.metric("🌳 Division Events", f"{total_divisions:,}")
    col4.metric("📄 Reports", sum(len(experiments.list_reports(e["id"])) for e in all_experiments))

    st.divider()

    st.subheader("Quick Actions")
    qa1, qa2, qa3 = st.columns(3)
    if qa1.button("＋ New Experiment", type="primary", use_container_width=True):
        _reset_new_experiment_wizard()
        _go("new")
    latest_with_report = next((e for e in done if experiments.latest_report(e["id"])), None)
    if qa2.button("📄 Open Last Report", use_container_width=True, disabled=latest_with_report is None):
        _go("detail", latest_with_report["id"])
    if qa3.button("📂 Browse Experiments", use_container_width=True):
        _go("list")

    st.divider()

    st.subheader("Recent Activity")
    recent_events = experiments.list_recent_events(limit=8)
    if not recent_events:
        st.caption("No activity yet — create an experiment to get started.")
    else:
        for ev in recent_events:
            label = experiments.EVENT_LABELS.get(ev["event_type"], ev["event_type"])
            line = f"✓ {ev['experiment_name']} — {label}"
            if st.button(line, key=f"activity_{ev['id']}"):
                _go("detail", ev["experiment_id"])
            st.caption(format_relative_time(ev["created_at"]))
    if len(all_experiments) > 0 and st.button("See all experiments →"):
        _go("list")


# ============================================================================
# My Experiments (list)
# ============================================================================
def render_list():
    st.title("My Experiments")
    if st.button("+ New Experiment", type="primary"):
        _reset_new_experiment_wizard()
        _go("new")

    st.divider()

    all_experiments = experiments.list_experiments()
    if not all_experiments:
        st.caption("No experiments yet.")
        return

    for e in all_experiments:
        dataset = experiments.get_dataset(e.get("dataset_id"))
        stats = experiments.get_experiment_stats(e["id"])  # None until this experiment has results

        with st.container(border=True):
            col_main, col_stats, col_action = st.columns([2.5, 2, 1])
            with col_main:
                st.markdown(f"**{e['name']}**")
                st.caption(_status_badge(e["status"]))
                st.write(f"Researcher: {e.get('researcher') or '—'}")
                st.write(f"Dataset: {dataset['filename'] if dataset else '—'}")
                st.write(f"Created: {format_relative_time(e['created_at'])}")
            with col_stats:
                if stats:
                    st.metric("Cells", stats["n_cells_detected"])
                    st.metric("Divisions", stats["n_divisions"])
                elif e["status"] in ("queued", "processing"):
                    st.caption("Still running — stats appear once it finishes.")
                elif e["status"] == "error":
                    st.caption("Run failed — no stats.")
                else:
                    st.caption("No results yet.")
            with col_action:
                n_reports = len(experiments.list_reports(e["id"]))
                st.caption(f"{n_reports} report(s)")
                if st.button("Open", key=f"open_{e['id']}", use_container_width=True):
                    _go("detail", e["id"])


# ============================================================================
# New Experiment — three real steps: Experiment Info -> Dataset -> Configuration -> Run
# ============================================================================
def render_new():
    st.title("New Experiment")

    if st.session_state.experiment_info is None:
        _render_wizard_progress(1)
        _render_new_step1_info()
    elif st.session_state.pending_dataset is None:
        _render_wizard_progress(2)
        _render_new_step2_dataset()
    else:
        _render_wizard_progress(3)
        _render_new_step3_configuration()


def _render_new_step1_info():
    st.subheader("1. Experiment Info")

    name = st.text_input("Experiment name", placeholder="e.g. Embryo Development Day 3")
    researcher = st.text_input("Researcher", placeholder="e.g. Dr. Ada Lovelace")
    st.caption("Both are optional — you can leave them blank and rename later.")

    if st.button("Continue →", type="primary"):
        st.session_state.experiment_info = {"name": name or None, "researcher": researcher or None}
        st.rerun()


def _render_new_step2_dataset():
    st.subheader("2. Dataset")
    info = st.session_state.experiment_info
    if info["name"]:
        st.caption(f"For: **{info['name']}**" + (f" · {info['researcher']}" if info["researcher"] else ""))

    examples_dir = PROJECT_ROOT / "examples"
    example_files = sorted(examples_dir.glob("*.tif")) if examples_dir.exists() else []
    chosen_example = None
    if example_files:
        st.caption("No data handy? Try an example:")
        cols = st.columns(len(example_files))
        for col, path in zip(cols, example_files):
            with col:
                if st.button(path.stem.replace("_", " ").title(), use_container_width=True):
                    chosen_example = path
        st.caption("Examples are synthetic placeholders (see examples/README.md), not real embryo data.")

    uploaded_file = st.file_uploader(
        "...or upload a .zip of a Zarr store, or a .tif/.tiff stack",
        type=["zip", "tif", "tiff"],
    )
    continue_clicked = st.button("Continue →", type="primary") if uploaded_file is not None else False

    if st.button("← Back"):
        st.session_state.experiment_info = None
        st.rerun()

    if chosen_example is not None or (continue_clicked and uploaded_file is not None):
        filename = chosen_example.name if chosen_example is not None else uploaded_file.name
        # Dataset is registered as its own row here, independent of any experiment — see
        # LabOS_Product_Spec_v1.md's reasoning: re-analyzing the same data with a different
        # Configuration shouldn't require re-uploading it.
        import uuid
        temp_dataset_dir = STORAGE_DIR / "_pending" / str(uuid.uuid4())

        with st.spinner("Reading data..."):
            if chosen_example is not None:
                volume = load_volume_from_bytes(chosen_example.name, chosen_example.read_bytes(), temp_dataset_dir)
            else:
                volume = load_volume_from_bytes(uploaded_file.name, uploaded_file.getvalue(), temp_dataset_dir)

        dataset_id = experiments.create_dataset(
            filename, shape=list(volume.shape), storage_path=str(temp_dataset_dir / filename)
        )

        st.session_state.pending_dataset = {
            "dataset_id": dataset_id,
            "is_example": chosen_example is not None,
        }
        st.session_state._volume_cache[dataset_id] = volume
        st.rerun()


def _render_new_step3_configuration():
    info = st.session_state.experiment_info
    pending = st.session_state.pending_dataset
    dataset = experiments.get_dataset(pending["dataset_id"])

    st.subheader("3. Configuration")
    st.write(f"**Dataset:** {dataset['filename']}  ·  shape {tuple(dataset['shape'] or [])}")
    if info["name"]:
        st.write(f"**Name:** {info['name']}")
    if info["researcher"]:
        st.write(f"**Researcher:** {info['researcher']}")

    saved_configs = experiments.list_configurations()
    options = ["Use defaults (from src/config.py)"] + [_config_summary(c) for c in saved_configs]
    # Pre-select whichever saved Configuration is marked default (Settings page's "Set as
    # Default"), so the common case of "just run it with my usual settings" is zero extra clicks.
    default_cfg = experiments.get_default_configuration()
    default_index = 0
    if default_cfg is not None:
        for i, c in enumerate(saved_configs):
            if c["id"] == default_cfg["id"]:
                default_index = i + 1
                break
    choice = st.selectbox("Configuration", options, index=default_index)
    if choice == options[0]:
        selected_config = None
    else:
        selected_config = saved_configs[options.index(choice) - 1]
    st.caption("Manage saved Configurations on the Settings page.")

    col1, col2 = st.columns(2)
    if col1.button("← Back", use_container_width=True):
        st.session_state.pending_dataset = None
        st.rerun()

    if col2.button("Run Analysis", type="primary", use_container_width=True):
        if selected_config is None:
            config_id = experiments.create_configuration("Defaults", **experiments.default_configuration_params())
        else:
            config_id = selected_config["id"]

        experiment_id = experiments.create_experiment(
            info["name"], researcher=info["researcher"],
            dataset_id=pending["dataset_id"], config_id=config_id,
        )

        # Move the dataset's raw file into the experiment's own folder (it was staged under
        # storage/_pending/ during step 2, before an experiment existed to own it).
        volume = st.session_state._volume_cache.pop(pending["dataset_id"], None)
        raw_dir = experiments.raw_dir(experiment_id)
        raw_dir.mkdir(parents=True, exist_ok=True)

        experiments.update_experiment(experiment_id, status="queued")
        (STORAGE_DIR / experiment_id / "status.json").write_text(
            json.dumps({"state": "queued", "message": ""})
        )
        is_example = st.session_state.get("pending_dataset", {}).get("is_example", False)


        thread = threading.Thread(
           target=process_job,
           args=(experiment_id, volume),
           kwargs={
               "use_cnn_filter": True,
               "config_id": config_id,
          },
          daemon=True,
        )
        thread.start()

        _reset_new_experiment_wizard()
        _go("detail", experiment_id)


# ============================================================================
# Settings — manage Configurations
# ============================================================================
def render_settings():
    st.title("Settings")
    st.caption("Detection/tracking parameters and physical calibration, saved as reusable, "
               "named Configurations — see LabOS_Product_Spec_v1.md's Persistent Storage "
               "section for why these are real database rows instead of only editable by "
               "hand-editing src/config.py.")

    st.subheader("New Configuration")
    defaults = experiments.default_configuration_params()
    with st.form("new_configuration"):
        config_name = st.text_input("Configuration name", placeholder="e.g. High Sensitivity")
        col1, col2 = st.columns(2)
        with col1:
            gaussian_sigma = st.number_input("Gaussian sigma", value=float(defaults["gaussian_sigma"]))
            detection_threshold = st.number_input("Detection threshold", value=float(defaults["detection_threshold"]))
            cell_radius = st.number_input("Cell radius (px)", value=float(defaults["cell_radius"]))
            tracking_max_distance = st.number_input("Tracking max distance (px)",
                                                       value=float(defaults["tracking_max_distance"]))
        with col2:
            division_max_distance = st.number_input("Division max distance (px)",
                                                       value=float(defaults["division_max_distance"]))
            pixel_size_um = st.number_input("Pixel size (µm) — 0 = uncalibrated", value=0.0)
            frame_interval_min = st.number_input("Frame interval (min) — 0 = uncalibrated", value=0.0)

        submitted = st.form_submit_button("Save Configuration", type="primary")
        if submitted and config_name:
            experiments.create_configuration(
                config_name,
                gaussian_sigma=gaussian_sigma,
                detection_threshold=detection_threshold,
                cell_radius=cell_radius,
                tracking_max_distance=tracking_max_distance,
                division_max_distance=division_max_distance,
                pixel_size_um=pixel_size_um or None,
                frame_interval_min=frame_interval_min or None,
            )
            st.success(f"Saved Configuration '{config_name}'.")
            st.rerun()

    st.divider()
    st.subheader("Saved Configurations")
    saved_configs = experiments.list_configurations()
    if not saved_configs:
        st.caption("No saved Configurations yet — new experiments use src/config.py's defaults.")

    for cfg in saved_configs:
        with st.container(border=True):
            title = f"**{cfg['name']}**"
            if cfg.get("is_default"):
                title += "  ⭐ Default"
            st.write(f"{title}  ·  {format_relative_time(cfg['created_at'])}")
            st.caption(_config_summary(cfg))

            renaming_key = f"renaming_{cfg['id']}"
            if st.session_state.get(renaming_key):
                new_name = st.text_input("New name", value=cfg["name"], key=f"rename_input_{cfg['id']}")
                col_save, col_cancel = st.columns(2)
                if col_save.button("Save name", key=f"rename_save_{cfg['id']}", type="primary"):
                    try:
                        experiments.rename_configuration(cfg["id"], new_name)
                        st.session_state[renaming_key] = False
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
                if col_cancel.button("Cancel", key=f"rename_cancel_{cfg['id']}"):
                    st.session_state[renaming_key] = False
                    st.rerun()
            else:
                col1, col2, col3, col4 = st.columns(4)
                if col1.button("Duplicate", key=f"dup_{cfg['id']}", use_container_width=True):
                    experiments.duplicate_configuration(cfg["id"])
                    st.rerun()
                if col2.button("Rename", key=f"ren_{cfg['id']}", use_container_width=True):
                    st.session_state[renaming_key] = True
                    st.rerun()
                if cfg.get("is_default"):
                    if col3.button("Unset default", key=f"undef_{cfg['id']}", use_container_width=True):
                        experiments.clear_default_configuration()
                        st.rerun()
                else:
                    if col3.button("Set as Default", key=f"def_{cfg['id']}", use_container_width=True):
                        experiments.set_default_configuration(cfg["id"])
                        st.rerun()
                confirm_key = f"confirm_delete_{cfg['id']}"
                if st.session_state.get(confirm_key):
                    if col4.button("⚠️ Confirm delete", key=f"del_confirm_{cfg['id']}", use_container_width=True):
                        try:
                            experiments.delete_configuration(cfg["id"])
                            st.session_state[confirm_key] = False
                            st.rerun()
                        except ValueError as e:
                            st.session_state[confirm_key] = False
                            st.error(str(e))
                else:
                    if col4.button("Delete", key=f"del_{cfg['id']}", use_container_width=True):
                        st.session_state[confirm_key] = True
                        st.rerun()


# ============================================================================
# Experiment Detail — Overview / Viewer / Analytics / Lineage / Reports / Exports
# ============================================================================
def render_detail():
    experiment_id = st.session_state.current_experiment_id
    exp = experiments.get_experiment(experiment_id)

    if exp is None:
        st.error("This experiment no longer exists.")
        if st.button("← Back to My Experiments"):
            _go("list")
        return

    st.title(exp["name"])
    st.caption(
        f"{_status_badge(exp['status'])}  ·  created {format_relative_time(exp['created_at'])}"
        f"  ·  id {experiment_id[:8]}"
    )

    tabs = st.tabs([
        "Overview", "Dataset", "Configuration", "Analysis History", "Timeline",
        "Viewer", "Analytics", "Lineage", "Reports", "Exports",
    ])
    status = get_status(experiment_id)
    has_results = experiments.has_results(experiment_id)

    dataset = experiments.get_dataset(exp.get("dataset_id"))
    config_row = experiments.get_configuration(exp.get("config_id"))

    # --- Overview: identity + live status + summary stats (Dataset/Configuration split out
    # into their own tabs below, per the v0.6.0 product review's "everything inside one
    # experiment" ask) ---
    with tabs[0]:
        st.subheader("Overview")
        st.write(f"**Name:** {exp['name']}")
        st.write(f"**Researcher:** {exp.get('researcher') or 'n/a'}")
        st.write(f"**Status:** {_status_badge(exp['status'])}")
        st.write(f"**Created:** {format_relative_time(exp['created_at'])}")
        st.write(f"**Last updated:** {format_relative_time(exp['updated_at'])}")

        if status["state"] in ("queued", "processing"):
            st.info(status.get("message") or "Working...")
            time.sleep(1.5)
            st.rerun()
        elif status["state"] == "error":
            st.error(f"Processing failed: {status['message']}")
        elif has_results:
            from src.analytics import compute_dataset_stats
            result = experiments.load_lineage_result(experiment_id)
            pixel_size_um = config_row.get("pixel_size_um") if config_row else None
            frame_interval_min = config_row.get("frame_interval_min") if config_row else None
            stats = compute_dataset_stats(result, pixel_size_um=pixel_size_um, frame_interval_min=frame_interval_min)
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Cells detected", stats["n_cells_detected"])
            col2.metric("Tracks", stats["n_tracks"])
            col3.metric("Divisions", stats["n_divisions"])
            col4.metric("Avg. velocity", f"{stats['avg_velocity']:.2f} {stats['avg_velocity_units']}")

            cached = get_cached_result(experiment_id)
            if cached is None:
                st.caption("Note: this experiment's raw data isn't held in memory this session "
                           "(e.g. after a server restart). The Lineage, Analytics, Reports, and "
                           "Exports tabs are all still fully available — the Viewer tab falls "
                           "back to pre-rendered images instead of a live interactive view, and "
                           "Analysis History's Re-run Analysis is disabled until this experiment "
                           "is re-run at least once in the current session.")

    # --- Dataset ---
    with tabs[1]:
        st.subheader("Dataset")
        if dataset:
            st.write(f"**Filename:** {dataset['filename']}")
            if dataset.get("shape"):
                st.write(f"**Shape (T, Z, Y, X):** {tuple(dataset['shape'])}")
            st.write(f"**Uploaded:** {format_relative_time(dataset['uploaded_at'])}")
            st.caption(
                "This dataset is a first-class record, independent of this one experiment — "
                "see LabOS_Product_Spec_v1.md — but starting a second experiment from the same "
                "data currently re-uploads it rather than reusing this row; that's a known gap, "
                "not the intended long-term behavior."
            )
        else:
            st.caption("No dataset on record for this experiment.")

    # --- Configuration ---
    with tabs[2]:
        st.subheader("Configuration")
        if config_row:
            st.write(f"**Name:** {config_row['name']}" + ("  ⭐ Default" if config_row.get("is_default") else ""))
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Gaussian sigma: {config_row['gaussian_sigma']}")
                st.write(f"Detection threshold: {config_row['detection_threshold']}")
                st.write(f"Cell radius (px): {config_row['cell_radius']}")
                st.write(f"Tracking max distance (px): {config_row['tracking_max_distance']}")
            with col2:
                st.write(f"Division max distance (px): {config_row['division_max_distance']}")
                st.write(f"Pixel size (µm): {config_row.get('pixel_size_um') or '0 = uncalibrated'}")
                st.write(f"Frame interval (min): {config_row.get('frame_interval_min') or '0 = uncalibrated'}")
        else:
            st.caption("This run used src/config.py's defaults directly (no saved Configuration).")
        st.caption("This is the Configuration behind the most recent run — see Analysis History "
                   "for what earlier runs used. Manage saved Configurations on the Settings page.")

    # --- Analysis History (v0.6.0) ---
    with tabs[3]:
        st.subheader("Analysis History")
        runs = experiments.list_analysis_runs(experiment_id)
        if not runs:
            st.caption("No analysis runs recorded yet.")
        else:
            for run in runs:
                run_config = experiments.get_configuration(run["config_id"])
                with st.container(border=True):
                    st.write(f"{_status_badge(run['status'])}  ·  started {format_relative_time(run['started_at'])}")
                    st.caption(f"Configuration: {_config_summary(run_config)}")
                    if run["status"] == "error" and run["error_message"]:
                        st.caption(f"Error: {run['error_message']}")
                    if run["report_id"]:
                        st.caption("Produced a report — see the Reports tab.")

        st.divider()
        st.subheader("Re-run Analysis")
        cached = get_cached_result(experiment_id)
        if cached is None:
            st.caption(
                "Can't re-run from here right now — this experiment's raw data isn't held in "
                "memory this session (e.g. after a server restart). This is the same v0 "
                "limitation noted on the Overview tab, not specific to Analysis History."
            )
        elif status["state"] in ("queued", "processing"):
            st.caption("An analysis is already running for this experiment.")
        else:
            saved_configs = experiments.list_configurations()
            options = ["Use current Configuration"] + [_config_summary(c) for c in saved_configs]
            choice = st.selectbox("Configuration for this run", options, key=f"rerun_cfg_{experiment_id}")
            if st.button("Re-run Analysis", type="primary", key=f"rerun_btn_{experiment_id}"):
                new_config_id = None if choice == options[0] else saved_configs[options.index(choice) - 1]["id"]
                experiments.update_experiment(experiment_id, status="queued")
                (STORAGE_DIR / experiment_id / "status.json").write_text(json.dumps({"state": "queued", "message": ""}))
                
                is_example = st.session_state.get("pending_dataset", {}).get("is_example", False)

                thread = threading.Thread(
                    target=process_job,
                    args=(experiment_id, volume),
                    kwargs={"use_cnn_filter": not is_example},
                    daemon=True,
                )
                thread.start()
                st.rerun()

    # --- Timeline (v0.6.0) ---
    with tabs[4]:
        st.subheader("Timeline")
        experiment_events = experiments.list_events(experiment_id)
        if not experiment_events:
            st.caption("No timeline events yet.")
        else:
            for ev in experiment_events:
                try:
                    time_label = datetime.fromisoformat(ev["created_at"]).strftime("%H:%M:%S")
                except ValueError:
                    time_label = ev["created_at"]
                label = experiments.EVENT_LABELS.get(ev["event_type"], ev["event_type"])
                st.write(f"**{time_label}** — {label}")
                if ev["message"]:
                    st.caption(ev["message"])

    # --- Viewer ---
    with tabs[5]:
        st.subheader("Viewer")
        if not has_results:
            st.caption("No results yet — see the Overview tab for status.")
        else:
            cached = get_cached_result(experiment_id)
            if cached is not None:
                try:
                    from src.interactive_viewer import build_plotly_figure
                    from src.analytics import compute_node_speeds

                    result = cached["result"]
                    pixel_size_um = config_row.get("pixel_size_um") if config_row else None
                    frame_interval_min = config_row.get("frame_interval_min") if config_row else None
                    speeds, units = compute_node_speeds(result, pixel_size_um, frame_interval_min)
                    n_frames = int(max((n.t for n in result.nodes), default=-1)) + 1
                    frame_idx = st.slider("Frame", 0, max(n_frames - 1, 0), 0, key=f"frame_{experiment_id}")
                    projection = volume[frame_idx].max(axis=0)
                    fig = build_plotly_figure(projection, result, frame_idx, speeds=speeds, speed_units=units)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Zoom/pan with the toolbar; hover a cell for its details.")
                except Exception as e:
                    st.warning(f"Interactive viewer unavailable ({e}); showing static view instead.")
                    cached = None

            if cached is None:
                results_dir = experiments.results_dir(experiment_id)
                frame_files = sorted((results_dir / "frames").glob("frame_*.png"))
                if frame_files:
                    frame_idx = st.slider("Frame", 0, len(frame_files) - 1, 0, key=f"frame_static_{experiment_id}")
                    st.image(str(frame_files[frame_idx]), use_container_width=True,
                             caption="Colored by track; white star marks a division "
                                     "(static — raw data not in memory this session)")

    # --- Analytics ---
    with tabs[6]:
        st.subheader("Analytics")
        if not has_results:
            st.caption("No results yet — see the Overview tab for status.")
        else:
            dashboard_path = experiments.results_dir(experiment_id) / "dashboard.png"
            if dashboard_path.exists():
                st.image(str(dashboard_path), use_container_width=True,
                          caption="Cell count, divisions, and velocity over the experiment")

    # --- Lineage ---
    with tabs[7]:
        st.subheader("Lineage")
        if not has_results:
            st.caption("No results yet — see the Overview tab for status.")
        else:
            cached = get_cached_result(experiment_id)
            persisted_result = cached["result"] if cached else experiments.load_lineage_result(experiment_id)

            all_track_ids = sorted(persisted_result.tracks.keys())
            choice = st.selectbox(
                "Highlight a cell's lineage (all descendants)",
                ["(none)"] + [f"track {tid}" for tid in all_track_ids],
                key=f"highlight_{experiment_id}",
            )
            highlight_track_id = int(choice.split()[1]) if choice != "(none)" else None

            import matplotlib.pyplot as plt
            from src.visualize import plot_lineage_tree
            fig_tree, ax_tree = plt.subplots(figsize=(9, max(4, len(persisted_result.tracks) * 0.35)))
            plot_lineage_tree(persisted_result, ax=ax_tree, highlight_track_id=highlight_track_id)
            st.pyplot(fig_tree)
            plt.close(fig_tree)

    # --- Reports (versioned) ---
    with tabs[8]:
        st.subheader("Reports")
        report_history = experiments.list_reports(experiment_id)
        if not report_history:
            st.caption("No reports generated yet — see the Overview tab for status.")
        else:
            st.caption(f"{len(report_history)} version(s) — most recent first. "
                       "Re-running analysis creates a new version rather than overwriting the last one.")
            for i, report in enumerate(report_history):
                report_pdf = report["storage_path"] / "report.pdf"
                label = "Latest" if i == 0 else f"Version from {format_relative_time(report['generated_at'])}"
                col1, col2 = st.columns([3, 1])
                col1.write(f"**{label}**  ·  {format_relative_time(report['generated_at'])}")
                if report_pdf.exists():
                    col2.download_button("Download", report_pdf.read_bytes(), file_name="report.pdf",
                                          key=f"report_dl_{report['id']}")
        st.caption("See the Reports page (sidebar) for every report across all experiments.")

    # --- Exports ---
    with tabs[9]:
        st.subheader("Exports")
        if not has_results:
            st.caption("No results yet — see the Overview tab for status.")
        else:
            results_dir = experiments.results_dir(experiment_id)
            tracks_csv = results_dir / "tracks.csv"
            ctc_csv = results_dir / "ctc_tracks.csv"
            if tracks_csv.exists():
                st.download_button("Download tracks.csv", tracks_csv.read_bytes(), file_name="tracks.csv")
            if ctc_csv.exists():
                st.download_button("Download ctc_tracks.csv (CTC format)", ctc_csv.read_bytes(),
                                    file_name="ctc_tracks.csv")


# ============================================================================
# Performance — real benchmark evidence (LabOS vs a baseline like TrackMate) on real
# datasets, recorded by scripts/run_ctc_benchmark.py / scripts/run_trackmate_comparison.py
# (v0.7.0 Epic 2)
# ============================================================================
def render_performance():
    st.title("Performance")
    st.caption(
        "Real validation evidence — LabOS scored against real dataset ground truth, alongside "
        "a baseline where available. Recorded by scripts/run_ctc_benchmark.py and "
        "scripts/run_trackmate_comparison.py, not editable from here."
    )

    all_runs = experiments.list_benchmark_runs()
    if not all_runs:
        st.caption(
            "No benchmark runs recorded yet. Run `python scripts/run_ctc_benchmark.py "
            "--dataset-dir <path> --sequence 01` against a real downloaded dataset — see "
            "RUNBOOK.md's Cell Tracking Challenge step — and its results will appear here."
        )
        return

    datasets_seen = []
    for run in all_runs:
        if run["dataset"] not in datasets_seen:
            datasets_seen.append(run["dataset"])

    for dataset_name in datasets_seen:
        dataset_runs = [r for r in all_runs if r["dataset"] == dataset_name]
        with st.container(border=True):
            st.subheader(dataset_name)
            labos_runs = [r for r in dataset_runs if not r["is_baseline"]]
            baseline_runs = [r for r in dataset_runs if r["is_baseline"]]

            cols = st.columns(max(len(dataset_runs), 1))
            for col, run in zip(cols, labos_runs + baseline_runs):
                with col:
                    label = run["method"] + (" (baseline)" if run["is_baseline"] else "")
                    st.markdown(f"**{label}**")
                    st.caption(format_relative_time(run["created_at"]))

                    def _fmt(value, suffix=""):
                        return f"{value:.3f}{suffix}" if value is not None else "not measured"

                    st.metric("Tracking accuracy", _fmt(run["adjusted_edge_jaccard"]))
                    st.metric("Division precision", _fmt(run["division_jaccard"]))
                    st.write(f"Edge precision / recall: {_fmt(run['edge_precision'])} / {_fmt(run['edge_recall'])}")
                    st.write(f"Runtime: {_fmt(run['runtime_seconds'], 's') if run['runtime_seconds'] is not None else 'not measured'}")
                    st.write(f"Memory: {_fmt(run['memory_delta_mb'], ' MB') if run['memory_delta_mb'] is not None else 'not measured'}")
                    if run["notes"]:
                        st.caption(run["notes"])

            if labos_runs and baseline_runs:
                labos_score = labos_runs[0]["adjusted_edge_jaccard"]
                baseline_score = baseline_runs[0]["adjusted_edge_jaccard"]
                if labos_score is not None and baseline_score is not None:
                    delta = labos_score - baseline_score
                    direction = "ahead of" if delta > 0 else ("behind" if delta < 0 else "tied with")
                    st.caption(
                        f"LabOS is {abs(delta):.3f} tracking-accuracy points {direction} "
                        f"{baseline_runs[0]['method']} on this dataset. This is one dataset, "
                        f"one run each — not a claim that generalizes on its own."
                    )

    st.divider()
    st.caption(
        "Every row here traces back to a real scored run against real ground truth — see "
        "docs/EVIDENCE_PACKAGE_TrackMate_Comparison.md for the fuller write-up format used "
        "for the incubator package, including what these numbers don't cover (manual "
        "intervention effort, report-generation quality)."
    )


# ============================================================================
# Reports Center — every report across every experiment (v0.6.0 Epic 3)
# ============================================================================
def render_reports_center():
    st.title("Reports")

    all_reports = experiments.list_all_reports()
    if not all_reports:
        st.caption("No reports generated yet.")
        return

    for report in all_reports:
        report_pdf = report["storage_path"] / "report.pdf"
        with st.container(border=True):
            col_main, col_dl = st.columns([3, 1])
            with col_main:
                st.markdown(f"**{report['experiment_name']}**")
                st.caption(f"PDF  ·  Generated {format_relative_time(report['generated_at'])}")
            with col_dl:
                if report_pdf.exists():
                    st.download_button(
                        "Download", report_pdf.read_bytes(),
                        file_name=f"{report['experiment_name']}_report.pdf",
                        key=f"center_dl_{report['id']}", use_container_width=True,
                    )

            col_preview, col_open, col_delete = st.columns(3)
            preview_key = f"center_preview_{report['id']}"
            if col_preview.button("Preview", key=f"center_prevbtn_{report['id']}", use_container_width=True):
                st.session_state[preview_key] = not st.session_state.get(preview_key, False)
            if col_open.button("Open Experiment", key=f"center_open_{report['id']}", use_container_width=True):
                _go("detail", report["experiment_id"])

            confirm_key = f"center_confirm_del_{report['id']}"
            if st.session_state.get(confirm_key):
                if col_delete.button("⚠️ Confirm delete", key=f"center_delconfirm_{report['id']}", use_container_width=True):
                    experiments.delete_report(report["id"])
                    st.session_state[confirm_key] = False
                    st.rerun()
            else:
                if col_delete.button("Delete", key=f"center_delbtn_{report['id']}", use_container_width=True):
                    st.session_state[confirm_key] = True
                    st.rerun()

            if st.session_state.get(preview_key) and report_pdf.exists():
                import base64
                b64 = base64.b64encode(report_pdf.read_bytes()).decode()
                st.markdown(
                    f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="600" '
                    f'style="border:1px solid #444;"></iframe>',
                    unsafe_allow_html=True,
                )


# ============================================================================
# Router
# ============================================================================
render_sidebar()

page = st.session_state.page
if page == "landing":
    render_landing()
elif page == "list":
    render_list()
elif page == "new":
    render_new()
elif page == "settings":
    render_settings()
elif page == "reports":
    render_reports_center()
elif page == "performance":
    render_performance()
elif page == "detail":
    render_detail()
else:
    render_landing()