# LabOS Product Spec v1

**Status: blueprint, not a record of what's built.** Where this spec describes something that
already exists in the code, it says so and cites the real file. Where it describes something
that doesn't exist yet, it says that too. The gap between the two sections is the actual
roadmap — not aspiration dressed up as progress.

## Table of Contents

- [Vision](#vision)
- [Target Users](#target-users)
- [Product Architecture](#product-architecture)
- [Core Modules](#core-modules)
- [Experiment Management — Data Model](#experiment-management--data-model)
- [Persistent Storage — Schema](#persistent-storage--schema)
- [User Workflow](#user-workflow)
- [MVP: Current State vs. This Spec](#mvp-current-state-vs-this-spec)
- [Version 1.0 Roadmap](#version-10-roadmap)

## Vision

LabOS is a workspace for live-cell imaging research: a researcher uploads a time-lapse
microscopy experiment and gets back tracked cells, detected divisions, lineage trees, and a
scientific report — without writing code, running ImageJ macros, or knowing what a Hungarian
assignment algorithm is. The underlying engine (`src/`) does the computer vision; LabOS is the
product surface that makes it usable by someone whose job is biology, not software.

Explicitly not the vision (yet): a general-purpose bioimage analysis platform. See
`docs/PHASE4_PITCH.md` for why leading with the narrow, provable thing (cell tracking) beats
leading with the platform story before there's evidence anyone wants the platform.

## Target Users

Carried over from `docs/STARTUP_REQUIREMENTS_DOCUMENT.md`, still `[HYPOTHESIS]` until Phase 3
interviews correct it:

- **Hands-on user** — PhD student / postdoc who personally processes the imaging data. Most
  likely daily user of LabOS specifically.
- **Decision-maker** — PI who doesn't run the analysis but decides what tools the lab adopts.
- **Intermediary** — core imaging facility staff, potentially serving many labs at once.

Which of these is the actual LabOS user vs. the actual buyer is still open — see
`docs/PHASE3_VALIDATION.md`.

## Product Architecture

```
LabOS
├── Dashboard
├── My Experiments
├── Experiment Details
│   ├── Overview
│   ├── Viewer
│   ├── Analytics
│   ├── Lineage
│   ├── Reports
│   └── Exports
└── Settings
```

Every future feature should belong somewhere in this tree. If it doesn't fit cleanly under one
of these six branches, that's a signal to either extend the architecture deliberately (and
update this document) or reconsider whether the feature belongs in LabOS at all — not to bolt
it on wherever it happens to fit today.

**Reality check:** the running app (`mvp/streamlit_app.py`) currently implements Dashboard, My
Experiments, and Experiment Details with five of these six tabs — Overview, Viewer, Analytics,
Lineage, Reports, Exports (six, not the "Overview/Viewer/Analytics/Reports/Exports" shown in
the original architecture sketch — Lineage was split out as its own tab in v0.3.0, see
`CHANGELOG.md`). **Settings does not exist yet.** See
[Roadmap](#version-10-roadmap).

## Core Modules

| Module | Purpose | Status |
|---|---|---|
| **Dashboard** | Landing page: hero section, experiment/report statistics, recent experiments, recent reports, entry point to create a new experiment | Built (`mvp/streamlit_app.py::render_landing`) |
| **My Experiments** | List of all experiments — name, status, relative time, open action | Built (`render_list`) |
| **Experiment Details → Overview** | Identity, dataset info, live processing status, summary stats once done | Built |
| **Experiment Details → Viewer** | Interactive per-frame view: zoom/pan, hover a cell for ID/parent/children/birth/death/track length/speed | Built, but the Plotly rendering itself has never executed in the environment that wrote it (see `RUNBOOK.md`) |
| **Experiment Details → Analytics** | Cell count, divisions, and velocity over time | Built |
| **Experiment Details → Lineage** | Lineage tree with "highlight this cell's descendants" | Built |
| **Experiment Details → Reports** | Download the combined scientific report (Experiment / Methods / Tracking Metrics / Division Analysis / Figures / Limitations / Appendix) | Built |
| **Experiment Details → Exports** | Download raw tracks.csv / CTC-format tracks.csv | Built |
| **Settings** | Physical calibration (pixel size, frame interval — see `src/config.py`'s `PIXEL_SIZE_UM`/`FRAME_INTERVAL_MIN`), detection/tracking parameter overrides, researcher profile | **Built.** Named, reusable Configurations, creatable/listable on the Settings page, prefilled from `src/config.py`'s real current defaults. |

## Experiment Management — Data Model

**Implemented** (`mvp/experiments.py`, `CREATE TABLE experiments`): `id`, `name`, `researcher`,
`dataset_id`, `config_id`, `status`, `created_at`, `updated_at` — all real, typed columns, not
a JSON blob. `update_experiment()` validates field names against this schema and raises on a
typo rather than silently accepting it (a real behavior change from an earlier version of this
table that used a flexible blob — see `CHANGELOG.md`).

| Field | Type | Notes |
|---|---|---|
| `id` | UUID | |
| `name` | text | e.g. "Embryo Development Day 3" |
| `researcher` | text | Free text, not a user account system — see rationale below |
| `dataset_id` | foreign key → `datasets` | |
| `status` | enum: created / queued / processing / done / error | |
| `created_at` | timestamp | |
| `updated_at` | timestamp | bumped on every `update_experiment()` call, verified distinct from `created_at` after a real update |
| `config_id` | foreign key → `configurations` | which detection/tracking parameters this run actually used — and actually used them, not just recorded them; see Roadmap |

Why `researcher` as free text and not a real user system: LabOS has no authentication and
Vision explicitly doesn't call for one yet — a lab of 3-5 people typing their own name is a
reasonable v1 answer; building real auth before there's evidence multiple labs are sharing one
deployment would be exactly the kind of premature infrastructure investment
`docs/PHASE2_MVP_ARCHITECTURE.md` already argued against for the job queue.

## Persistent Storage — Schema

**Implemented:** four SQLite tables (`experiments`, `datasets`, `configurations`, `reports`),
verified as a real SQLite file (checked its magic bytes and queried it independently) in
`tests/test_experiments.py`. Still SQLite on purpose ("SQLite is enough" continues to be true
at this scale — the trigger for moving to something heavier is concurrent multi-lab usage, not
a milestone number):

```sql
CREATE TABLE experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    researcher TEXT,
    dataset_id TEXT REFERENCES datasets(id),
    config_id TEXT REFERENCES configurations(id),
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE datasets (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    shape TEXT,              -- JSON: [T, Z, Y, X]
    uploaded_at TEXT NOT NULL,
    storage_path TEXT NOT NULL
);

CREATE TABLE reports (
    id TEXT PRIMARY KEY,
    experiment_id TEXT REFERENCES experiments(id),
    generated_at TEXT NOT NULL,
    storage_path TEXT NOT NULL
);

CREATE TABLE configurations (
    id TEXT PRIMARY KEY,
    gaussian_sigma REAL,
    detection_threshold REAL,
    cell_radius REAL,
    tracking_max_distance REAL,
    division_max_distance REAL,
    pixel_size_um REAL,
    frame_interval_min REAL
);
```

**Why split these out instead of leaving them as files-by-convention**, one real product
decision per table, not a blanket "more structure is better":

- **`datasets`** as its own table matters if the same uploaded dataset should ever be reusable
  across multiple experiments (e.g. re-running with different parameters without re-uploading).
  Today's 1-experiment-to-1-dataset assumption is simple but means re-analyzing the same data
  means re-uploading it. Worth deciding deliberately, not by default.
- **`reports`** as its own table matters if reports should be versioned (re-generate after a
  re-run, keep the old one) rather than overwritten in place, which is today's behavior.
- **`configurations`** as its own table is what makes the Benchmark Dashboard
  (`src/benchmark.py`) and a future Settings page connect to real experiments — "which
  parameters produced this result" becomes a query, not something only reconstructible from
  `src/config.py`'s state at some unknown past moment (this is a real, current limitation:
  `src/report.py`'s Appendix section only shows the values `config.py` had *when the report was
  generated*, with no persisted record if those values later change).

## User Workflow

```
Dashboard
    ↓
New Experiment (name + researcher)
    ↓
Upload Dataset
    ↓
Run Analysis
    ↓
Viewer
    ↓
Analytics
    ↓
Generate Report
    ↓
Export
```

**Implemented, then extended in v0.6.0:** `mvp/streamlit_app.py::render_new` originally split
this into two real steps (Dataset, then Configuration+Run), then a further product review
(v0.6.0 Epic 1) asked for a visible 3-step wizard — Step 1 collects name/researcher (Experiment
Info), Step 2 registers the Dataset (a real row, independent of any experiment), Step 3 shows a
Configuration picker (saved Configurations from Settings, or "Use defaults", pre-selecting
whichever is marked default) and only starts `process_job()` once "Run Analysis" is clicked.
This was sequenced deliberately both times: splitting the steps only became worth doing once
Settings/Configurations existed to actually choose between at that pause point — see the
original reasoning preserved below.

The reasoning that drove that sequencing, kept here since it's still the right way to think
about this kind of UX decision when it comes up again elsewhere in the product:

- **Keep it bundled:** fewer clicks, matches the "one workflow" MVP discipline from
  `docs/PHASE2_MVP_ARCHITECTURE.md`.
- **Split them:** lets a researcher upload data, then choose or adjust analysis parameters
  before committing to a run — only relevant once there's something real to choose between.

## MVP: Current State vs. This Spec

What's real today, mapped against this document, so "MVP" means something checkable rather than
a vibe. Updated after actually implementing the v1.0 Roadmap below, not just planning it —
every ✅ here has a corresponding test in `tests/test_experiments.py`:

| Spec requirement | Status |
|---|---|
| Dashboard, My Experiments, Experiment Details (Viewer/Analytics/Lineage/Reports/Exports) | ✅ Built and proven end-to-end against real pipeline output (see `RUNBOOK.md`) |
| Persistent storage (experiments survive a restart) | ✅ Built — SQLite, verified with an independent connection |
| Experiment has a name | ✅ |
| Experiment has a researcher | ✅ Real column on `experiments`, settable at creation |
| Experiment references a reusable dataset entity | ✅ `datasets` table — a `storage_path`-verified real row, not a filename string |
| Experiment references a recorded configuration | ✅ `configurations` table — immutable per-experiment snapshot, and **actually affects the analysis**: `mvp/pipeline.py` passes a Configuration's parameters into `CellDetector`/`HungarianTracker`/`LineageBuilder`, verified by a test that runs the same data through two different Configurations and checks the detected cell counts genuinely differ |
| Settings module | ✅ Create/list named Configurations, prefilled from `src/config.py`'s real current defaults; v0.6.0 added Duplicate/Rename/Delete (refuses if an experiment still references it)/Set as Default |
| Distinct "Run Analysis" step | ✅ New Experiment is now three steps — Experiment Info, Dataset, then Configuration+Run Analysis (choose a saved Configuration or use defaults) |
| Reports versioned per experiment | ✅ `reports` table — each `process_job()` run creates a new row + folder; verified that re-running doesn't touch a previous version's files |
| Re-analyze an experiment with a different Configuration, without losing history | ✅ v0.6.0: `analysis_runs` table — one row per actual run (Configuration used, status, report produced); Experiment Detail's Analysis History tab lists them and can trigger a re-run, gated on the raw volume still being cached this session (same limitation as the Viewer tab) |
| Per-experiment activity timeline | ✅ v0.6.0: `events` table, shown in Experiment Detail's Timeline tab — logs only the pipeline's actual stages (created, dataset uploaded, analysis started, tracking finished, report generated, or analysis failed), not invented sub-steps |
| Cross-experiment "what's happening" view | ✅ v0.6.0: Dashboard's Recent Activity feed reads the same `events` table as Timeline — one log, two views |
| Reports Center (every report, across every experiment, in one place) | ✅ v0.6.0: new sidebar page, `experiments.list_all_reports()` — Download / Preview (inline PDF) / Delete |

**Three real bugs found while building this, not just asserted away:**
1. SQLite returns `REAL` columns as Python `float`s — `cell_radius=12` becomes `12.0`, which
   crashed `skimage.peak_local_max` (needs an `int` for `min_distance`). Fixed by an explicit
   `int(...)` cast at the one place it mattered, not a blanket type change everywhere.
2. `LineageResult.to_dataframe()` on zero detections produced a DataFrame with **zero columns**,
   not just zero rows — writing that to CSV made a file `pandas.read_csv` couldn't even parse
   back. Reachable in practice the moment a Configuration is strict enough to detect nothing.
   Fixed with an explicit `columns=` argument.
3. `src/analytics.py` originally read calibration from `config.py`'s global module state —
   harmless for one notebook at a time, but a genuine race condition waiting to happen once
   multiple experiments with different Configurations can run concurrently in background
   threads. Refactored to pass calibration through as explicit function arguments instead, and
   tested that two different calibrations computed back-to-back don't leak into each other.

## Version 1.0 Roadmap

In dependency order — each item unlocks the next rather than being independently prioritizable.
**Status: implemented**, updated from the original planning-only version of this section:

1. ✅ **Extend the `experiments` table** with `researcher` (free text, no auth system) and
   `updated_at`. Smallest possible change, immediately closes the biggest Experiment Management
   gap from Step 3.
2. ✅ **Add the `datasets` table**, decoupling "uploaded file" from "experiment." Requires deciding
   the reuse question above before writing the migration, not after.
3. ✅ **Add the `configurations` table**, and a minimal Settings page that can edit the physical
   calibration fields (`PIXEL_SIZE_UM`, `FRAME_INTERVAL_MIN`) that already exist in
   `src/config.py` but aren't reachable outside editing source code. This is the single most
   concrete, already-scoped gap in the current product.
4. ✅ **Split "Upload Dataset" from "Run Analysis"** as separate steps, now that there's something
   (a chosen configuration) to select in between them.
5. ✅ **Add the `reports` table** and stop overwriting reports in place on re-run.

Deliberately not on this list: anything from `docs/PHASE4_PITCH.md`'s platform vision
(segmentation, colony analysis, etc.) — that's a different roadmap, for after this one and for
after Phase 3 evidence, not before.

## What's Still Not Verified

Same standing caveat as every prior version: `streamlit run mvp/streamlit_app.py` itself has
not executed in the environment that built this (`streamlit` isn't installed there). Every
piece of underlying *logic* — the two-step wizard, Settings creating real Configuration rows,
a Configuration actually changing detection output, report versioning across re-runs — was run
end-to-end against the real pipeline (see `tests/test_experiments.py` and this repo's
`CHANGELOG.md`), but the actual Streamlit rendering needs a real machine to confirm.
