# LabOS — Research Workspace (v0)

Read [`docs/PHASE2_MVP_ARCHITECTURE.md`](../docs/PHASE2_MVP_ARCHITECTURE.md) first — this is
the "get in front of a real researcher this week" version, not the production one.

## What this is now

A hero landing page (sidebar for navigation) → **My Experiments** (cards: name, researcher,
status, dataset, per-experiment cell/division counts, report count) → **New Experiment** as
three real wizard steps (Experiment Info → Dataset → Configuration — pick a saved Configuration
or use defaults) → **Experiment Detail** (Overview / Dataset / Configuration / Analysis History
/ Timeline / Viewer / Analytics / Lineage / Reports / Exports tabs) → **Settings**
(create/manage named Configurations: Duplicate, Rename, Delete, Set as Default) → **Reports**
(every report across every experiment: Download, Preview, Delete). See
[`LabOS_Product_Spec_v1.md`](../LabOS_Product_Spec_v1.md) for the full blueprint.

Backed by `mvp/experiments.py` — a real six-table SQLite database (`labos.db`: `experiments`,
`datasets`, `configurations`, `reports`, `events`, `analysis_runs`), not a hand-rolled JSON
index and not a single flat table either. A Configuration genuinely changes the analysis (see
`mvp/pipeline.py`'s `config_params` handling) — it's not just recorded metadata. Reports are
versioned: re-running an experiment creates a new row and folder, never overwriting a previous
run's files. An experiment can be re-run with a different Configuration from its own Analysis
History tab, without losing the record of what earlier runs used — each run is its own
`analysis_runs` row; every notable moment along the way is an `events` row, which is what backs
both the per-experiment Timeline tab and the Dashboard's cross-experiment Recent Activity feed.
Create an experiment today, close the browser, come back tomorrow: it's still there.

## Run it

```bash
pip install -r ../requirements.txt
pip install -e ..    # from mvp/, or `pip install -e .` from the repo root
streamlit run streamlit_app.py
```

Needs `models/best_model.pth` and `models/norm_stats.npz` to already exist (train a model
first — see the main README's Usage section) for the full CNN-filtered pipeline. Without those,
use `scripts/smoke_test.py --no-cnn` to verify everything else works first (see `RUNBOOK.md`).

## What actually persists across a restart, and what doesn't

Verified in the sandbox that built this (see `tests/test_experiments.py`, including a test
that reloads the experiments module fresh — simulating a real process restart, not just
reading the same in-memory objects — and a separate test that opens `labos.db` with an
independent `sqlite3` connection to confirm it's a genuine SQLite file, magic bytes and all):

- **Persists:** the experiment list itself (name, status, created date — all in `labos.db`),
  the lineage tree (with "highlight this cell's descendants" — reconstructed from
  `tracks.csv`, not from memory), all analytics, and every file under Reports/Exports.
- **Does NOT persist (v0 scope):** the raw uploaded volume itself. This means the *live*
  per-frame interactive Plotly viewer (hover for details, hover during zoom/pan) only works
  within the session that processed the data — after a restart, the Viewer tab falls back to
  the static frame images `src/report.py` already rendered at processing time, which do
  persist. The Overview tab tells you which mode you're in rather than silently degrading.
  Persisting the raw volume too is a reasonable v1 upgrade once there's evidence it's worth
  the storage cost (see `docs/PHASE2_MVP_ARCHITECTURE.md`).

## Status: core logic proven, Streamlit rendering itself unverified

Every piece of *logic* behind this app — experiment creation, status transitions, the full
New Experiment → Detail data flow, and specifically the post-restart fallback path — was
actually run end-to-end against real pipeline output while building this (not just asserted).
What's **not** verified: `streamlit run streamlit_app.py` itself, because `streamlit` isn't
installed in the sandbox that wrote this. Same standing caveat as the Plotly interactive
viewer. Run it and expect to fix small UI things — most likely candidates:

- The Zarr-zip extraction assumes a single top-level array/group; adjust if a real upload's
  structure differs.
- The TIFF axis-order assumption (`T, Z, Y, X`) is a guess — real acquisition software may
  order axes differently, and tifffile can read that from the file's metadata instead of
  guessing, once you have a real sample file to check.
- Nothing here handles concurrent uploads gracefully yet (fine for one researcher at a time,
  not fine for two people using it simultaneously — the in-memory `_RESULT_CACHE` and
  background-thread job processing in `mvp/pipeline.py` are both v0-scale, not production
  infrastructure).

## What "done" looks like for v0

Not "production-ready" — just: you can hand a lab a link, they create an experiment, upload a
video, get tracks + a division count + a report back, and it's still there when they come back
— without you sitting next to them explaining Python. That's the whole bar for a Phase 3
validation conversation.
