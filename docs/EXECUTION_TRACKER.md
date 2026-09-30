# Execution Tracker

Six workstreams, running in parallel, not sequential phases. This file gets checked off with
real dates and real links — a checkbox with no evidence behind it is exactly the kind of
"planning theater" this tracker exists to prevent. Update it as things actually happen, not
in a batch before a review.

## Workstreams

### 1. Product
**Goal:** a researcher uploads data, gets results, downloads them — without anyone from the team
helping.
- [x] Detection → tracking → division detection → visualization → report, proven end-to-end on
  a real file (`mvp/make_test_data.py` + current pipeline validation) —
  found and fixed two real bugs in the process. **CNN filtering step still unverified** (needs
  `torch` + a real environment — see `RUNBOOK.md` step 3).
- [x] Visualization upgraded from raw red dots to colored per-track trajectories, division
  markers, a lineage tree, and a combined `report.pdf` (`src/visualize.py`, `src/report.py`)
- [x] Interactive zoom/pan/hover-to-inspect viewer built (`src/interactive_viewer.py`,
  Plotly-based) — data layer tested and proven against real pipeline output; the actual Plotly
  rendering has never executed (plotly not installable in the building environment) — needs
  `RUNBOOK.md` step 4 on a real machine to confirm
- [x] Analytics dashboard (cell count/divisions/velocity over time, per-track summaries) built
  and proven against real pipeline output (`src/analytics.py`) — with honest pixels-vs-real-unit
  handling (see `src/config.py`'s `PIXEL_SIZE_UM`/`FRAME_INTERVAL_MIN`, both unset by default)
- [ ] Tested against at least one real (or realistic) microscopy file, not synthetic data
- [ ] An outside person (not you) completes the full workflow unassisted

### 2. Science
**Goal:** quantitative answers, not "it seemed to work."
- [x] Scoring metric implemented (`src/evaluate.py`, 14 tests) — the competition's actual
  edge Jaccard / adjusted edge Jaccard / division Jaccard, not an approximation. **This is a
  scorer, not a score** — still needs real ground-truth data run through it.
- [x] Benchmark harness implemented (`src/benchmark.py`) with a naive-baseline comparator and
  a `Method | Tracking Score | Runtime` table generator, demonstrated end-to-end on synthetic
  data. **Still a harness, not a benchmark** — see the README's Benchmark section for exactly
  what's real vs. demonstrated-on-synthetic-data here.
- [x] Real-data benchmark workflow established (`scripts/run_ctc_benchmark.py` + `scripts/run_trackmate_comparison.py` + `scripts/load_ctc_ground_truth.py`) - current v0.7 workflow uses public CTC data and documented benchmark-v2 evidence.
- [ ] Detection accuracy measured against ground truth on real (not synthetic) data
- [ ] Tracking + division accuracy measured via `src/evaluate.py` on real ground truth
- [ ] Runtime + memory usage measured for at least one realistically-sized volume
- [ ] Comparison run against at least one incumbent tool (TrackMate is the easiest first
  comparison — free, scriptable, widely used; `src/benchmark.py`'s `load_trackmate_xml()` is
  ready but unverified against a real export — see `RUNBOOK.md` step 5) on the same data

### 3. Customer Discovery
**Goal:** five *real* interviews, per `PHASE3_VALIDATION.md`.
- [ ] Interview 1 — name/role: _____ — date: _____ — signal: weak / medium / strong
- [ ] Interview 2 — name/role: _____ — date: _____ — signal: weak / medium / strong
- [ ] Interview 3 — name/role: _____ — date: _____ — signal: weak / medium / strong
- [ ] Interview 4 — name/role: _____ — date: _____ — signal: weak / medium / strong
- [ ] Interview 5 — name/role: _____ — date: _____ — signal: weak / medium / strong
- [ ] Synthesis table filled in (`PHASE3_VALIDATION.md`)

### 4. Business
**Goal:** a reasoned answer to "who pays, and how much" — not a guess.
- [ ] Buyer identified (individual lab / core facility / pharma) — informed by *which persona in
  the SRD actually showed strong signal in Phase 3*, not decided independently of it
- [ ] Pricing model hypothesis stated (per-analysis / subscription / open-core) and why
- [ ] At least one real pricing reaction from a Phase 3 conversation (did anyone flinch, did
  anyone ask "how much" unprompted — that's a data point, a guess isn't)

### 5. Legal/IP
**Goal:** a stated position, not silence.
- [ ] Open-source vs. proprietary vs. open-core decided, with reasoning (the SRD's Business
  Model section has a starting hypothesis — confirm or revise it)
- [ ] Data handling policy for uploaded microscopy data exists (even a one-paragraph draft) —
  needed before any real lab uploads real, possibly-unpublished data
- [ ] A decision on whether any part of this is patentable / worth publishing as a paper instead
  (these can be in tension — publishing discloses; patenting requires not having disclosed yet)

### 6. Incubator Package
**Goal:** only assembled once 1–5 have real outputs to point to.
- [ ] Pitch deck filled in from `PHASE4_PITCH.md`'s outline, with every `[FROM PHASE 3]`
  placeholder replaced by real data — checked against that document's honesty checklist before
  it goes anywhere

## Suggested six-week pass (adjust dates, keep the sequencing logic)

The logic that matters more than the exact week numbers: **Science and Customer Discovery need
to inform Business before Business gets written**, and **Incubator Package is last, on purpose**.

| Week | Focus | Deliverable |
|---|---|---|
| 1 | Product | MVP actually runs, tested on real/realistic data |
| 2 | Science | Benchmark report (accuracy, runtime, comparison vs. TrackMate) |
| 3 | Customer Discovery | 5 real interviews completed |
| 4 | Product + Business | Product changes based on what interviews revealed; first real business-model draft |
| 5 | Legal/IP | Open-source/proprietary decision + data policy drafted |
| 6 | Incubator Package | Pitch deck assembled from real workstream outputs |
