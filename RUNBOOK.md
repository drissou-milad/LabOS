# Runbook — actually running this project

This is what to run, in order, to do the things that genuinely needed your machine and your
real data — the ones flagged throughout this project as "I can't do this in the sandbox."
Each step says plainly what's already verified vs. what's still on you to confirm.

## 0. Environment

```bash
git clone <your repo>
cd biohub-cell-tracking-repo
pip install -r requirements.txt
pip install -e .          # makes `src` and `mvp` importable everywhere, no sys.path hacks
```

Set `DATASET_PATH` in `src/config.py` to wherever you have the competition data.

## 1. Run the real test suite

Every test in this project has been executed as raw Python assertions (since `pytest` itself
wasn't installable in the sandbox that built this) — never through the actual `pytest` binary.
That's a real gap. Close it:

```bash
pip install -r requirements-dev.txt
pytest -v
```

**63 tests should pass** (17 original + 12 lineage + 14 evaluate + 16 visualize/report/benchmark
+ 4 mvp pipeline). If any fail, that's genuinely useful signal — dependency version drift
(NumPy/SciPy/PyTorch API changes) is exactly the kind of thing that only shows up at real
execution time, and I told you as much every time I couldn't run them for real. Report back
anything that fails.

## 2. Historical MVP smoke test

**None of that is the same as "tested on Windows."** Please actually run the current v0.7 benchmark and application workflows on a Windows machine before considering them verified.

Current v0.7 validation is based on public Cell Tracking Challenge data and the benchmark workflow documented in **step 9**.
## 3. Run the real MVP, with the real trained model

```bash
streamlit run mvp/streamlit_app.py
```

This is **LabOS**: a hero landing page (sidebar for navigation) → **My Experiments** (name,
researcher, status, relative time) → **New Experiment** as two steps — Step 1 uploads
`mvp/test_data/synthetic_test.tif` from step 2 (or a gallery example) and collects name/
researcher; Step 2 picks a Configuration (try creating one on the **Settings** page first, or
use defaults) and starts the run — → **Experiment Detail** page with tabs (Overview / Viewer /
Analytics / Lineage / Reports / Exports). Create an experiment, close the tab, reopen `streamlit
run mvp/streamlit_app.py` — it should still be in "My Experiments," and its lineage tree /
analytics / reports should all still work (the live per-frame Plotly view specifically will
fall back to static images after a real restart — see `mvp/README.md` for exactly what persists
and why). Try re-running the same experiment with a different Configuration (e.g. a much higher
detection threshold on the Settings page) and confirm the Overview tab's cell count actually
changes — that's the test that caught a real SQLite int/float bug while building this (see
`CHANGELOG.md`'s v0.5.0 entry). Experiment metadata now lives in a real SQLite file with four
tables, `mvp/storage/labos.db` — open it with any SQLite browser (or `sqlite3
mvp/storage/labos.db "SELECT * FROM experiments"`, `"... FROM configurations"`, etc.) to see it
directly.

This is the one thing in the whole project I genuinely cannot verify myself — no `torch`, no
browser, no Streamlit in the sandbox. Every piece of *logic* behind it (experiment creation,
status transitions, the full data flow, the restart-fallback path specifically, the SQLite file
itself opened independently with a separate connection) was run end-to-end against real
pipeline output while building it — see `tests/test_experiments.py` and `CHANGELOG.md`'s
v0.4.0 entry — but the actual Streamlit rendering needs your machine.

## 4. Interactive viewer, dashboard, and gallery — some of this unverified

```bash
streamlit run mvp/streamlit_app.py
```

Look for: the example gallery buttons at the top of step 1 (synthetic placeholders — see
`examples/README.md`), real zoom/pan on the frame view (Plotly's toolbar), hovering a cell
shows its ID/parent/children/birth/death frame/track length/speed, a "highlight this cell's
lineage" selector that fades everything except the selected cell and its descendants (both on
the frame view and the lineage tree), and a dashboard image (cell count, divisions, velocity
over time) below the tracked view. **`plotly` isn't installable in the sandbox that wrote
this**, so `src/interactive_viewer.py`'s actual rendering function has never executed — it's
written against Plotly's documented API, and the data it feeds to Plotly (`build_hover_data()`,
descendant highlighting) is fully tested and proven correct against real pipeline output
(including a real two-generation division), but the rendering call itself needs your machine to
confirm. The lineage-tree highlighting is matplotlib-based (`st.pyplot`), not Plotly, and is
low-risk by comparison — that part is the same proven code path as everything else in
`src/visualize.py`. If the Plotly viewer errors, the app falls back to the fully-proven static
image view automatically rather than breaking the page — but check the actual Plotly view
works, don't just accept the fallback silently.

## 5. Historical memory profiling

The former `scripts/profile_memory.py` workflow measured RSS on the small synthetic MVP file. The resulting measurement was dominated by fixed report-generation overhead and was not representative of real multi-GB microscopy volumes.

This script is retained as historical/local tooling but is not used as a v0.7 validation claim.
## 6. Windows compatibility

I can't run anything on an actual Windows machine - this sandbox is Linux, and there's no way around that. What I *could* do: a static review for platform-specific bugs. Found and fixed one real one: `src/config.py`'s `DATASET_PATH` defaulted to a hardcoded Windows-only path (`D:\Datasets\...`) with no override - this would hard-fail on Linux/Mac with no clear error.
Fixed to read `BIOHUB_DATASET_PATH` from the environment first (see `CHANGELOG.md`). Also checked for: `os.path` string-concatenation (none - `pathlib` used throughout), hardcoded `/` in path construction (none found), `os.fork`/Unix-only syscalls (none), `chmod`/symlinks (none).

**None of that is the same as "tested on Windows."** Please actually run `pytest`, `streamlit run mvp/streamlit_app.py`, and the current v0.7 benchmark workflow on a Windows machine before considering the Windows execution path verified.

## 7. Real public-data validation (v0.7)

The previous `run_real_benchmark.py` workflow is historical and is not part of the current v0.7 validation path.

For reproducible real-data validation, use the public Cell Tracking Challenge workflow in **step 9**:
- `scripts/load_ctc_ground_truth.py`: load CTC ground truth
- `scripts/run_ctc_benchmark.py`: run LabOS against the public dataset
- `scripts/run_trackmate_comparison.py`: compare LabOS with TrackMate on the same dataset
- `scripts/benchmark_v2/`: benchmark, audit, and reproducibility tooling

## 9. Run LabOS on a real public dataset (v0.7)

The Kaggle BioHub dataset used during earlier development is real, but the current v0.7 reproducibility workflow uses a public benchmark dataset.
The [Cell Tracking Challenge](https://celltrackingchallenge.net/) (CTC) provides public benchmark datasets for cell tracking and is the basis of the current reproducible validation workflow.
**Pick a small dataset first** — don't start with a multi-GB one:
- [Fluo-N2DL-HeLa](https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip)
  — 2D, HeLa cells, has divisions, moderate size. Good first choice.
- [Fluo-N2DH-GOWT1](https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip)
  (53 MB) — smaller, good for a fast first pass.
- Full list: [2D+time datasets](https://celltrackingchallenge.net/2d-datasets/) and
  [3D+time datasets](https://celltrackingchallenge.net/3d-datasets/) (3D ones are much larger —
  stay 2D for your first real run).

Download and unzip one. You'll get a folder like `Fluo-N2DL-HeLa/` containing `01/`, `02/`
(raw image sequences), `01_GT/`, `02_GT/` (ground truth — the `TRA` subfolder is what matters
here; `SEG` is segmentation-only and not used by this pipeline).

```bash
python scripts/inspect_ground_truth.py  # if you haven't already — sanity habit, not required for CTC
python scripts/load_ctc_ground_truth.py path/to/Fluo-N2DL-HeLa/01_GT/TRA
```

The second command should print real node/edge/division counts for that sequence. If it
errors, the dataset's folder layout doesn't match what `scripts/load_ctc_ground_truth.py`
assumes — read the error message, it names the exact file it expected.

Then run LabOS against it for real:

```bash
python scripts/run_ctc_benchmark.py --dataset-dir path/to/Fluo-N2DL-HeLa --sequence 01
```

Start with `--frame-limit 10` on your very first run of a new dataset — faster feedback on
whether the wiring works before committing to a full-length run. This records LabOS's numbers
into `labos.db`'s `benchmark_runs` table automatically; open the app and check the Performance
page (sidebar) — your real numbers should be there. Add `--trackmate-xml
path/to/export.xml --trackmate-runtime-seconds N` once you've also run this same sequence
through Fiji/TrackMate (see step 10 below), to get both rows on the same Performance card.

`scripts/load_ctc_ground_truth.py` is tested against a small hand-authored fixture
(`tests/fixtures/ctc_sample/`), same honest caveat as `load_trackmate_xml()` in step 10: this
proves the parser implements the CTC schema correctly, not that it survives every real
dataset's quirks untested. If a specific CTC dataset trips it up, that's real signal, not a
sign you're doing something wrong.

## 10. TrackMate comparison (optional, needs Fiji)

1. Install [Fiji](https://fiji.sc/), open your volume, run TrackMate, export the model as XML.
2. `from src.benchmark import load_trackmate_xml; nodes, edges = load_trackmate_xml("your_export.xml")`
3. Feed those into `src.benchmark.score_method()` alongside the other methods.

`load_trackmate_xml()` is written against TrackMate's documented XML schema. `tests/test_benchmark.py`
now checks it against a hand-authored fixture (`tests/fixtures/trackmate_sample.xml`) that follows
that same documented schema — this proves the parser correctly implements the schema it claims to,
including the int node-ID cast (a real class of bug: TrackMate's IDs would otherwise come back as
strings, silently failing to match this project's own int node IDs if you ever compare the two
directly). It does **not** prove a real Fiji export looks exactly like the fixture — TrackMate
version differences, extra attributes, or edge cases in real data could still break it. Expect to
debug it against your first real export; that's a one-time cost, not a recurring one.

## 11. External researcher validation

Use the validation materials in `docs/PHASE3_VALIDATION.md`, `docs/RESEARCHER_INTERVIEW_GUIDE.md`, and `docs/RESEARCHER_FEEDBACK.md` to collect structured feedback from external researchers.

This step is intentionally outside the codebase: it is needed to assess scientific relevance, workflow fit, and practical limitations beyond the reproducible technical validation already documented in v0.7.

## Summary: v0.7 validation status

| Area | Status |
|---|---|
| Public Cell Tracking Challenge benchmark workflow | [done] Established and documented |
| CTC ground-truth loading and benchmark tooling | [done] Public/reproducible scripts committed |
| TrackMate baseline methodology and artifacts | [done] Baseline artifacts committed; real external researcher/lab validation still pending |
| benchmark-v2 distance-gated evaluation | [done] Established and documented |
| Run13 held-out detector evaluation | [done] Completed with reproducible evidence package |
| Run13 preprocessing / normalization audit | [done] Documented, including unlabeled sample-level test normalization |
| Run13 spatial / temporal audit | [done] Targeted audits completed and documented |
| Detector interpretation | [caution] Measurable ranking/discriminative signal; not a production operating point |
| Biological validation | [pending] External researcher validation pending |
| Laboratory validation | [pending] Real laboratory validation pending |
| Production readiness | [pending] Not claimed at v0.7 |
