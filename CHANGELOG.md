# Changelog

## v0.7.0 — Research Validation: real public datasets, in-app Performance page, feedback + demo prep

Reframes the milestone as "Research Validation, not more features" (the review's own words).
Two of the four objectives are genuinely code (real CTC dataset support, an in-app benchmark
page); two are deliberately NOT code (researcher interviews, a rehearsed demo) — given
structure and templates here, not automated, because they can't honestly be automated.

### Added
- **`scripts/load_ctc_ground_truth.py`**: converts a real [Cell Tracking Challenge](https://celltrackingchallenge.net/)
  (CTC) format ground-truth folder (`man_track*.tif` label masks + `man_track.txt` lineage) into
  the same `(nodes, edges)` shape every other evaluation path in this project uses — the mirror
  operation of `src/lineage.py::LineageResult.to_ctc_tracks()`, which already writes LabOS's own
  results in this exact format. `load_ctc_image_sequence()` alongside it stacks a CTC raw-image
  sequence into LabOS's own `(T, Z, Y, X)` volume shape. Tested against a small, hand-authored
  fixture (`tests/fixtures/ctc_sample/`) built to the real documented CTC schema — 6 new tests
  in `tests/test_load_ctc_ground_truth.py`, verified down to exact centroid coordinates and
  exact division edges, not just "didn't crash." Same honest caveat as the TrackMate importer:
  this proves the parser implements the schema correctly, not that every real CTC dataset's
  quirks are covered — no real dataset was available to test against in this sandbox (no
  network access here).
- **`scripts/run_ctc_benchmark.py`**: runs LabOS's real pipeline against a real downloaded CTC
  dataset, scores it against the real ground truth, optionally scores a real TrackMate export
  against the same ground truth too, and — new, versus the earlier
  `scripts/run_trackmate_comparison.py` — **records every row into LabOS's own database**
  (`mvp/experiments.py::record_benchmark_run()`) rather than only printing to a terminal.
  Verified end-to-end (ground truth loading → volume loading → real detection/tracking →
  scoring → DB write) against a synthetic stand-in dataset built in the CTC folder layout,
  since no real CTC data could be downloaded in this sandbox.
- **`benchmark_runs` table** in `mvp/experiments.py`'s schema: `record_benchmark_run()` /
  `list_benchmark_runs()` / `delete_benchmark_run()`. Unmeasured metrics (e.g. a baseline run
  where memory wasn't profiled) are stored as `NULL`, never fabricated as `0` — same convention
  `src/benchmark.py::score_external_method()` already established for `runtime_seconds`.
- **Performance page** (`mvp/streamlit_app.py::render_performance`, new sidebar link): shows
  every recorded benchmark run, grouped by dataset, LabOS's numbers next to any baseline's, with
  an explicit one-line "ahead of / behind" comparison and a caveat that one dataset/one run
  isn't a claim that generalizes. Verified against both empty state and real recorded data
  using the same Streamlit-stub method introduced in v0.6.0.
- **`docs/RESEARCHER_INTERVIEW_GUIDE.md`**: who to talk to (varied roles, not five of the same),
  the actual question list in order, and explicit "don't argue with a negative answer" /
  "don't lead the witness" guidance — the failure modes that make interview evidence
  unconvincing to an incubator.
- **`docs/RESEARCHER_FEEDBACK.md`**: fill-in template for all 5 interviews plus a summary table,
  so the five conversations turn into one document you can actually hand someone.
- **`docs/DEMO_SCRIPT.md`**: a timed, second-by-second 5-minute demo script matching the exact
  flow requested (Create → Upload → Configure → Run → Viewer → Lineage → PDF → Export), a
  pre-demo checklist (including a pre-done fallback experiment in case a live run is slow), and
  a "what to do if it breaks live" section.

### Verified
- Full test suite (`tests/test_experiments.py` + `tests/test_load_ctc_ground_truth.py`, 58
  tests) run as raw assertions in the sandbox, all passing.
- `scripts/run_ctc_benchmark.py` run end-to-end against a synthetic dataset built in real CTC
  folder layout (raw image sequence + `man_track.txt` + label-mask TIFFs), confirming the full
  chain works, including the `benchmark_runs` DB write — not just that each piece compiles.

### Deliberately not done
- No actual CTC dataset was downloaded or run in this environment — this sandbox has no
  network access. The loader and benchmark script are built, tested against a schema-accurate
  fixture, and wired into the app; running them against a real download is on you, per
  `RUNBOOK.md`'s new step 9.
- No researcher interviews were conducted and no demo was rehearsed — both are structured with
  templates/scripts above, but doing them is real human work no amount of code substitutes for.

## v0.6.0 — Experiment Workspace, Analysis History, Timeline, Report Center

Five epics from a second product-review round, aimed at deepening the workflow rather than
adding more top-level menus (per the review's own framing). The two architecturally real ones
— Analysis History and Timeline — needed a genuine new data model, not just UI reshuffling;
everything else builds on top of it.

### Added
- **Two new tables in `mvp/experiments.py`'s SQLite schema**: `events` (one row per notable
  moment — experiment created, dataset uploaded, analysis started, tracking finished, report
  generated, or analysis failed) and `analysis_runs` (one row per actual `process_job()` run,
  recording which Configuration it used, its status, and which report it produced). Both are
  the shared foundation behind three of the five epics below, not duplicated per-feature logs.
  `log_event()` / `list_events()` / `list_recent_events()` and `create_analysis_run()` /
  `update_analysis_run()` / `list_analysis_runs()`, plus `list_all_reports()` / `delete_report()`
  for the Report Center. 12 new tests in `tests/test_experiments.py`; full suite (52 tests) run
  as raw assertions in the sandbox and confirmed passing.
- **Epic 1 (Experiment Workspace):** Experiment Detail is now 10 tabs — Overview / Dataset /
  Configuration / Analysis History / Timeline / Viewer / Analytics / Lineage / Reports /
  Exports — up from 6, with Dataset and Configuration split out of the old combined Overview.
  Analysis History lists every run and, when the raw volume is still cached this session, lets
  you re-run the SAME experiment with a DIFFERENT saved Configuration without losing the
  record of earlier runs. Verified end-to-end against the real pipeline: two runs, two
  different configs, two `analysis_runs` rows, two report versions, and the experiment's
  "current" config correctly following the latest run while history stays intact.
- **Epic 2 (Better Dashboard):** the old Experiments/Completed/Running/Reports-only Recent
  Experiments+Recent Reports columns are replaced with a single Recent Activity feed, reading
  the same `events` table as Timeline.
- **Epic 3 (Report Center):** new sidebar page (`render_reports_center`) listing every report
  across every experiment — Download, inline Preview (base64-embedded PDF, no extra
  dependency), Delete (with a confirm step), Open Experiment.
- **Epic 4 (Research Timeline):** per-experiment chronological event log, new tab. Deliberately
  logs only the pipeline's actual distinct stages, not the finer-grained steps in the original
  product review's mockup ("Lineage reconstructed", "Analytics exported") that don't correspond
  to separate stages `mvp/pipeline.py` actually has — see "Deliberately not done" below.
- **Epic 5 (Landing copy):** "LabOS — The Research Operating System" headline and the
  "Manage... analyze... reconstruct... generate..." positioning copy.

### Changed
- `mvp/pipeline.py::process_job()` gained an optional `config_id` parameter (defaults to the
  experiment's existing config_id, so every prior call site is unaffected) and now logs events
  and creates/updates an `analysis_runs` row at each stage, including the error path.
- `mvp/experiments.py::create_experiment()` now logs its own `experiment_created` (and
  `dataset_uploaded`, if a dataset_id was given) events automatically, so no call site can
  forget to log experiment creation.
- `mvp/experiments.py::delete_experiment()` now also clears `events` and `analysis_runs` rows
  for the deleted experiment (previously only `experiments` + `reports`).

### Verified
- Regenerated the entire flow against real data (not mocks): `experiments.create_experiment`,
  `pipeline.process_job` (twice, with two different Configurations), then every `render_*`
  function in `mvp/streamlit_app.py` — including all 10 Experiment Detail tabs and the new
  Reports Center — executed against a minimal Streamlit stub built for this purpose, to catch
  `NameError`/`AttributeError` bugs that plain `ast.parse` can't see. This is not a substitute
  for actually clicking through it in a browser — do that too.

### Deliberately not done
- Timeline/Recent Activity event types match `mvp/pipeline.py`'s actual stages (analysis
  started → tracking finished → report generated) rather than the original mockup's finer
  breakdown ("Lineage reconstructed", "Analytics exported") — those aren't separable stages in
  the current pipeline (lineage reconstruction happens inside the single detection+tracking
  call), and inventing event types that don't correspond to real pipeline stages would make
  the Timeline decorative rather than accurate.
- "Re-run Analysis" only works while the raw volume is cached in memory this session — same v0
  limitation as the Viewer tab (raw uploaded volumes aren't persisted to disk yet). Disabled
  with an explanation rather than silently failing or faking a re-run.
- Dataset reuse across experiments (Dataset tab notes this) is still not implemented — starting
  a second experiment from the same data currently re-uploads it. Flagged, not fixed, in scope.

## v0.5.3 — Product review round: real stats, cards, wizard, config management, branding

Implements the four items picked from a design review of running screenshots (Dashboard,
Sidebar, My Experiments, New Experiment, Settings, Branding): Dashboard/My Experiments now
show real numbers instead of zeros, My Experiments is card-based, New Experiment is a real
3-step wizard, Configurations support Duplicate/Rename/Delete/Set as Default, and branding
uses the uploaded logo with updated positioning copy.

### Added
- `mvp/experiments.py`: `duplicate_configuration()`, `rename_configuration()`,
  `delete_configuration()` (refuses if any experiment still references the config — the whole
  point of Configurations being immutable history breaks if one can vanish out from under a
  past run), `set_default_configuration()` / `clear_default_configuration()` /
  `get_default_configuration()` (a new `is_default` column, migrated in automatically via
  `ALTER TABLE` for anyone with an existing `labos.db` — verified against a simulated
  pre-migration database, not just a fresh one), and `get_experiment_stats()` (reuses
  `src/analytics.compute_dataset_stats()` against the same persisted `tracks.csv` the
  Experiment Detail Overview tab already reads, so the numbers can't drift between the two).
  11 new tests in `tests/test_experiments.py`, all run as raw assertions in the sandbox and
  confirmed passing, including the migration one.
- `mvp/streamlit_app.py`: Dashboard now sums real Cells Tracked / Division Events across every
  completed experiment (an experiment still queued/processing/errored contributes nothing,
  not a misleading zero) plus a Quick Actions row (New Experiment / Open Last Report / Browse
  Experiments); My Experiments is now bordered cards (name, status, researcher, dataset,
  per-experiment cell/division counts, report count, Open); New Experiment is a real 3-step
  wizard (Experiment Info → Dataset → Configuration) with a progress indicator, replacing the
  previous 2-step flow that bundled name/researcher into the upload step; Settings' saved
  Configurations gained Duplicate / Rename / Delete (with a confirm step) / Set as Default
  controls, and New Experiment's step 3 now pre-selects whichever Configuration is marked
  default.
- `mvp/assets/logo.png`: the molecular-mark logo, now shown in the sidebar in place of the 🧬
  emoji heading. Landing page copy updated to "The Operating System for Modern Biological
  Research" / the "Analyze… Track… Reconstruct… Generate…" framing. Sidebar caption changed
  "Research Workspace" → "Research OS".

### Fixed
- `requirements.txt` was missing `streamlit` entirely, despite it being the whole product's
  UI framework — added. (It evidently worked for at least one real install already, likely
  from a pre-existing global/other install; still a real gap for a genuinely clean machine.)

### Deliberately not done
- No fabricated "Welcome back, Dr. Smith"-style greeting on the Dashboard — there's no login/
  identity system behind who's using LabOS right now, and inventing a name would be a fake
  personalization, not a real one. Worth building for real once auth exists.

## v0.5.2 — Combined LabOS-vs-TrackMate comparison script + evidence package template

Priority 4's next actionable piece: once you have a real TrackMate export, this removes the
manual work of stitching LabOS's own timed run and TrackMate's scored output into one table.

### Added
- `scripts/run_trackmate_comparison.py`: runs LabOS's real detector + tracker + lineage on the
  same sample used for ground truth (timed for real, not synthetic), loads a real TrackMate XML
  export via `load_trackmate_xml()`, scores both against the same ground truth with the same
  tolerance, and prints one combined table. Explicitly does NOT run TrackMate itself — that's a
  Fiji/Java plugin, has to happen on your machine by hand — and says so in its own docstring.
  Reuses `scripts/run_real_benchmark.py`'s `load_ground_truth()` rather than duplicating it, so
  a fix there doesn't need a second fix here.
- `scripts/__init__.py` so the above import works cleanly.
- `docs/EVIDENCE_PACKAGE_TrackMate_Comparison.md`: a fill-in-the-blanks template covering every
  field the incubator asked for (dataset, runtime, accuracy, division detection, manual
  intervention, report generation) plus an explicit "if TrackMate wins a row, write that" rule
  and a raw-output appendix, so the eventual write-up is traceable back to real numbers.

### Verified
- `run_trackmate_comparison.py` statically checked (AST parse, call signatures matched against
  `src/benchmark.py`'s actual function signatures) — NOT executed end-to-end, since it needs
  `torch`/`zarr` (unavailable in this sandbox) and the real competition dataset. Needs your
  machine; see `RUNBOOK.md`.

## v0.5.1 — TrackMate importer hardened with schema-fixture tests; stale README roadmap fixed

Reality check against `LabOS_Product_Spec_v1.md`'s 5-priority list: priorities 1–3 (end-to-end
workflow, real experiment management, persistent storage) turned out to already be fully
implemented as of v0.5.0 — the README's Roadmap section just hadn't been updated to say so.
Priority 4 (LabOS vs TrackMate benchmark evidence) is the one genuinely open item, and it
needs a real machine with Fiji and the real competition dataset, neither available here — see
`RUNBOOK.md` steps 7–8. This release does the part of priority 4 that *could* be done without
either of those: making `load_trackmate_xml()` more trustworthy before it ever sees a real file.

### Added
- `tests/fixtures/trackmate_sample.xml`: a hand-authored fixture following TrackMate's
  documented XML model-export schema (two tracks, one division, 10 spots across 4 frames —
  mirrors `tests/test_benchmark.py`'s existing synthetic dataset shape for direct comparison).
  Explicitly NOT a real Fiji export (none available in this environment) — exists to test the
  parser against the schema it claims to implement, not to simulate real-world export quirks.
- Four new tests in `tests/test_benchmark.py` covering `load_trackmate_xml()`: correct node/edge
  parsing, correct division structure (one spot with two outgoing edges), and — most
  importantly — that the parsed result actually scores correctly through `src/evaluate.py`
  (perfect precision/recall/division-Jaccard when scored against itself), not just that parsing
  doesn't crash. Ran all four as raw assertions in the sandbox (no `pytest` binary available
  here either) and confirmed they pass.

### Fixed
- `load_trackmate_xml()` now casts Spot IDs and Edge source/target IDs to `int` instead of
  leaving them as the raw XML attribute strings. Every other node source in this codebase
  (`LineageResult.to_dataframe()`, `run_real_benchmark.py`'s ground truth) uses int node IDs;
  a TrackMate row with string IDs would still "run" without erroring (node_id is just an opaque
  dict key to `src/evaluate.py`) but would be a silent footgun the moment anyone tries to
  cross-reference a TrackMate node ID against this project's own node IDs directly.
- README.md's Roadmap section: moved the Streamlit dashboard, experiment management, and
  persistent-storage bullets from "Not yet implemented" (stale since v0.4.0/v0.5.0) to "Done,"
  and replaced the generic "run benchmark against real ground truth" bullet with the specific,
  actionable one — a real LabOS-vs-TrackMate number needs the real dataset + a real Fiji
  export, not more code; `RUNBOOK.md` already has the exact steps.
- `RUNBOOK.md` step 8 and its summary table: updated to describe the new fixture-based test
  coverage and what it does and doesn't prove.



Implements `LabOS_Product_Spec_v1.md`'s Version 1.0 Roadmap in full — not just the plan, the
actual schema and UI. Following its dependency order exactly.

### Added
- `mvp/experiments.py`'s SQLite schema expanded from one table to four: `experiments` gained
  `researcher` and `updated_at` (bumped on every update, verified distinct from `created_at`);
  new `datasets` table (real rows, `storage_path` required — an `IntegrityError` if you try to
  register one without knowing where its file lives); new `configurations` table (immutable,
  named, per-experiment snapshots — deliberately NOT deduplicated, so a later edit can never
  retroactively change what an earlier experiment says it used); new `reports` table (each
  analysis run gets its own row + folder, verified that a second run never touches the first
  version's files).
- **Configurations are a real feature, not metadata**: `mvp/pipeline.py`'s
  `run_detection_and_tracking()` now accepts `config_params` and passes them into
  `CellDetector`/`HungarianTracker`/`LineageBuilder`. Verified with a test that runs identical
  data through two Configurations (a normal detection threshold vs. an absurdly strict one) and
  checks the detected cell counts genuinely differ — not just that the code runs.
- `mvp/streamlit_app.py`: New Experiment is now two real steps (Upload Dataset, then Run
  Analysis — pick a saved Configuration or use `src/config.py`'s defaults); a new Settings page
  creates/lists named Configurations; Overview tab shows researcher, dataset, and the actual
  Configuration used; Reports tab shows full version history with a download button per version,
  not just the latest.
- `src/analytics.py` and `src/report.py`: calibration (`pixel_size_um`, `frame_interval_min`)
  now threaded through as explicit function arguments end to end (`compute_node_speeds` →
  `compute_dataset_stats` → `plot_dashboard` → `generate_report` → `_limitations_text`), so a
  report reflects the *specific experiment's* Configuration, not whatever `src/config.py`'s
  global state happened to be when the report was generated.

### Fixed — three real bugs, found by actually running this, not by review
1. **SQLite type coercion crashed the detector.** `REAL` columns come back as Python `float`
   (`cell_radius=12` becomes `12.0`), and `skimage.peak_local_max`'s `min_distance` requires an
   `int` — raised a `TypeError` the first time a Configuration was actually used. Fixed with an
   explicit `int(...)` cast at the one call site that needed it.
2. **Zero detections produced an unreadable CSV.** `pd.DataFrame([])` has zero *columns*, not
   just zero rows — `LineageResult.to_dataframe()` on an empty result wrote a CSV file
   `pandas.read_csv()` couldn't parse back at all (`EmptyDataError`), crashing
   `experiments.load_lineage_result()`. Reachable in practice the moment a Configuration is
   strict enough to detect nothing — which is exactly what happened while testing Configuration
   #1 above. Fixed with an explicit `columns=` argument in both `to_dataframe()` and
   `to_ctc_tracks()`.
3. **A latent concurrency bug in `src/analytics.py`.** It originally read `PIXEL_SIZE_UM`/
   `FRAME_INTERVAL_MIN` from `config.py`'s global module state — harmless for one notebook at a
   time, but a real race condition waiting to happen once multiple experiments with different
   Configurations can run concurrently in `mvp/pipeline.py`'s background threads. Refactored
   into an explicit `_resolve_calibration()` helper; verified with a test that computes two
   different calibrations back-to-back and confirms neither leaks into the other (and that an
   uncalibrated call right after isn't contaminated either).

### Added (tests)
- 25 new tests (152 total, up from 127; 140 actually executed and passing in this environment).

## v0.4.0 — SQLite storage, hero landing page, sidebar navigation, 6-tab experiment detail

### Changed
- `mvp/experiments.py`'s storage backend swapped from a hand-rolled JSON index to a real
  SQLite database (`mvp/storage/labos.db`) — same public API, so nothing downstream needed to
  change beyond the tests. Verified as genuinely SQLite (not just API-compatible) by opening
  the file with a completely independent `sqlite3` connection and checking its magic bytes.
- `mvp/streamlit_app.py`: added a persistent sidebar (LabOS branding + Dashboard / My
  Experiments / New Experiment navigation, replacing in-page "← Back" buttons as the primary
  nav); landing page rebuilt as a hero section ("AI-powered workspace for live-cell imaging
  research. Analyze microscopy experiments. Track cells. Detect divisions. Generate scientific
  reports.") with a Statistics section (experiments / completed / running / reports counts);
  My Experiments list now shows relative time ("3 hours ago") instead of a raw ISO timestamp;
  Experiment Detail's tabs consolidated from 7 to the requested 6 (Overview / Viewer /
  Analytics / Lineage / Reports / Exports) — Dataset info and Analysis Status both folded into
  Overview, and Lineage (with descendant highlighting) split out into its own tab.
- `mvp/formatting.py` (new) — `format_relative_time()`, the logic behind "3 hours ago." A real
  bug was caught while testing this, not just asserted away: the first version used a 30-day
  threshold before switching from "N days ago" to "N weeks ago," so a 21-day-old experiment
  would say "21 days ago" instead of the expected "3 weeks ago" — found by testing an actual
  21-day case rather than only the bucket boundaries, and fixed by moving the threshold to 7
  days, matching normal relative-time UX conventions.

### Branding audit (Task 1)
Did a full repo-wide grep for "BioHub" and classified every occurrence. Result: no changes were
needed this version — the LabOS/BioHub split (LabOS for the product everywhere; "BioHub Cell
Tracking" kept only for genuine Kaggle-competition references in `notebooks/`, `kaggle/`,
`src/dataset.py`'s `BioHubDataset` class, and `config.py`'s dataset path) was already correctly
in place from earlier work this session. Verified rather than assumed — see the actual grep
output categorized in this version's development notes.

### Added
- 20 new tests (127 total, up from 117; 115 actually executed and passing in this environment).

## v0.3.0 — LabOS: persistent multi-experiment workspace

### Added
- `mvp/experiments.py` — a persistent experiment registry (JSON index + one folder per
  experiment on disk, no database — the same "simplest thing that works" call as the rest of
  this MVP). `create_experiment()`, `list_experiments()`, `get_experiment()`,
  `update_experiment()`, `load_lineage_result()`. Verified against a *real* simulated restart
  (reloading the Python module fresh mid-test, not just re-reading in-memory objects) — an
  experiment created before the "restart" is still there, with the correct status, after it.
- `LineageResult.from_dataframe()` (`src/lineage.py`) — reconstructs a full result (nodes,
  edges, divisions, tracks) from `to_dataframe()`'s own CSV output. This is what makes
  persistence actually work for the *interactive* parts (lineage tree, descendant highlighting)
  after a restart, not just the static report files — verified with a real CSV write-then-read
  round trip against a two-generation division.
- `mvp/streamlit_app.py` rewritten as **LabOS**: Landing (experiment/report counts, recent
  experiments, recent reports, "New Experiment") → **My Experiments** (list, most recent first)
  → **New Experiment** (name + upload/gallery, same logic as before) → **Experiment Detail**
  (Overview / Dataset / Analysis Status / Viewer / Analytics / Reports / Exports tabs).
- `mvp/pipeline.py`'s `process_job()` now syncs status into the experiment registry at each
  stage, not just the polling `status.json`.
- 13 new tests (117 total, up from 104; 105 actually executed and passing in this environment).

### Honesty note carried through this version
The raw uploaded volume is **not** persisted past the processing session (v0 scope) — only the
CSV-derived lineage result is. This means the live per-frame Plotly viewer (needs actual image
data) only works within the session that processed it; after a restart it falls back to
pre-rendered static frame images, which *do* persist. The lineage tree with descendant
highlighting works either way, since it only needs `tracks.csv`. The Experiment Detail page's
Analysis Status tab says which mode you're in rather than silently degrading — this was a
deliberate design decision, not an oversight, and is documented in `mvp/README.md`.

Every piece of *logic* behind this (experiment creation, status transitions, the full New
Experiment → Detail data flow, and specifically the post-restart fallback path) was run
end-to-end against real pipeline output while building this. `streamlit run
mvp/streamlit_app.py` itself has not executed — `streamlit` isn't installed in the environment
that wrote this, same standing caveat as the Plotly viewer since v0.2.0.

## v0.2.0 — Descendant highlighting, expanded benchmark dashboard, dataset gallery, scientific report

### Added
- `LineageResult.get_descendant_track_ids()` / `get_descendant_node_ids()` — BFS through
  divisions, verified against a real two-generation (grandchild) division, not just a
  single-level case. Wired into `plot_lineage_tree()`'s `highlight_track_id` and
  `interactive_viewer.build_plotly_figure()`'s `highlight_track_id` (data layer tested; Plotly
  rendering unverified, same standing caveat) — a "highlight this cell's whole lineage"
  selector in the MVP.
- `src/benchmark.py` expanded: real RSS memory measurement per method run (psutil), precision
  and recall (derived from the same tp/fp/fn the competition score already computes — not
  separately invented), a "node detection rate" in place of an undefined "accuracy",
  `score_external_method()` so a real TrackMate export can appear in the same dashboard table
  without pretending this harness measured its runtime, and `plot_benchmark_dashboard()` (one
  page, four charts: runtime, memory, precision/recall, tracking score).
- `examples/` — a dataset gallery (Priority 5). **Synthetic placeholders, not real embryo
  data** — this project has no access to real microscopy files. Verified genuinely
  distinguishable through the real pipeline (crowded has more detected cells than sparse;
  division-rich produces an actual detected division; sparse doesn't) rather than just
  differently named. Two real bugs found and fixed while building this: cells drifting out of
  the canvas bounds over multiple frames, and `hash(string) % 1000` used as a "seed," which is
  not actually reproducible across Python process restarts (string hashing is randomized by
  default) — replaced with fixed integer seeds.
- `src/report.py` restructured into Experiment → Methods → Tracking Metrics → Division Analysis
  → Figures → Limitations → Appendix (Priority 6). Limitations section is grounded in the
  actual current implementation (detection threshold not recalibrated per-dataset, 3D reduced
  to max-projection, division detection is a distance heuristic not a learned model, no motion
  prediction, calibration-dependent units) — not a generic disclaimer. Appendix includes the
  real config values used and a raw track-summary table. `generate_report()` gained an optional
  `benchmark_rows` parameter to embed the full Method-comparison dashboard when available.
- 25 new tests (104 total, up from 79; 92 of them — everything except `test_model`/
  `test_dataset`/`test_seed`, which need `torch`/`zarr` — actually executed and passing).

## v0.1.0 — Phase 1 MVP complete

Version note: this was briefly bumped to `1.0.0` in an earlier pass, at explicit request, before
any real user had touched the product. That was a mistake, corrected here — `0.x.y` is the
honest semver signal for "unvalidated, API can still change," which is where this project
actually is. See `docs/EXECUTION_TRACKER.md` for what "validated" would actually require.

### Added
- `src/lineage.py` — `LineageBuilder`: division detection + CTC-style persistent track IDs,
  built on `HungarianTracker` without modifying it
- `src/evaluate.py` — the competition's exact scoring metric (edge Jaccard, adjusted edge
  Jaccard, division Jaccard), transcribed from Biohub's published spec
- `src/visualize.py` — colored per-track trajectories, division markers, lineage tree drawing
- `src/report.py` — ties evaluation, visualization, and CSV export into one `report.pdf`,
  including a grounded Method section built from actual run config (not templated prose)
- `src/analytics.py` — per-node speed, per-track summaries (birth/death/parent/children/track
  length/avg speed), dataset-level dashboard stats, and three charts (cell count over time,
  divisions over time, velocity histogram) — all honest about pixels/frame vs. real units,
  see below
- `src/interactive_viewer.py` — Plotly-based interactive frame viewer: real zoom/pan (via
  Plotly's default toolbar) and hover-to-inspect (ID, parent, children, birth/death frame,
  track length, speed). **Not executed in the environment that wrote it** — plotly wasn't
  installable there; verify before trusting it (see RUNBOOK.md)
- `src/benchmark.py` — comparison harness (ours vs. a naive greedy baseline, plus a TrackMate
  XML importer, also unverified against a real export)
- `mvp/` — a Streamlit v0 MVP (upload -> process -> interactive track -> download), plus
  `mvp/pipeline.py` (the actual logic, importable without `streamlit`), `mvp/make_test_data.py`
  (synthetic smoke-test data), and `scripts/smoke_test.py` (headless end-to-end proof)
- 79 tests total (up from 17 at the start of this MVP push); 67 of them (everything except
  `test_model`/`test_dataset`/`test_seed`, which need `torch`/`zarr`) actually executed and
  passing as of this version — see RUNBOOK.md for the real-`pytest` caveat

### Fixed
- `src/config.py`'s `DATASET_PATH` defaulted to a hardcoded Windows-only path
  (`D:\Datasets\...`) with no way to override it — would hard-fail on Linux/Mac/CI. Now reads
  `BIOHUB_DATASET_PATH` from the environment first, falling back to the original hardcoded
  value for backward compatibility. Found via a Windows/Linux portability review (this project
  has never actually been run on Windows by the tooling that built it — see RUNBOOK.md).
- `experiments/*/norm_stats.npz` was never promoted to `models/` alongside the checkpoint,
  meaning inference silently fell back to less-accurate per-image normalization by default
- Model class name (`CellClassifierCNN` vs. the real `CellCNN`), checkpoint path/format
  mismatches, a broken `NearestNeighborTracker` import that didn't exist, and assorted
  duplicated/dead code across the original notebooks — see earlier README history / git log for
  the full list from the initial engineering-review pass

### Known limitations at this version
- `LineageBuilder` is not yet wired into `src/predict.py`, `06_pipeline.ipynb`, or
  `kaggle_submission.ipynb` — those still do frame-to-frame tracking only
- No real benchmark number exists yet — `scripts/run_real_benchmark.py` is ready but
  unexecuted (needs the real dataset + `scripts/inspect_ground_truth.py`'s schema check first)
- The CNN filtering step (`torch`) has never been executed in the environment that built this —
  everything downstream of it (tracking, division detection, visualization, report) has been
  proven working on a real file; the CNN step itself has not
- Zero real users. Zero Phase 3 interviews conducted. See `docs/EXECUTION_TRACKER.md`.
