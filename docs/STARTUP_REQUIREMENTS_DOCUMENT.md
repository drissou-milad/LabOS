# Startup Requirements Document (SRD)

**Status: living document, last substantive update — pre-Phase-3.** Sections marked
`[HYPOTHESIS]` are founder assumptions, not confirmed facts, and should be overwritten with real
data as the six workstreams below produce it — not treated as settled just because they're
written down in complete sentences. A document that reads as authoritative but rests on
unvalidated assumptions is more dangerous than an obviously-incomplete one; keeping the
`[HYPOTHESIS]` tags visible is intentional, not a formatting afterthought.

This is the constitution. When a product, business, or pitch decision needs making, it should
be checked against this document — and this document should change when evidence contradicts it,
not the other way around.

## Mission

Give researchers doing developmental/cell biology the same automated cell-tracking capability
that currently requires either hours of manual annotation or specialized computational biology
skills most wet labs don't have in-house.

## Vision

`[HYPOTHESIS]` A future where quantitative image analysis (tracking, and eventually
segmentation, counting, colony/tissue analysis) is as easy for a biology lab to use as uploading
a file — not because the underlying models are simple, but because the product hides that
complexity. Whether this actually becomes a multi-module platform or stays a focused tool
depends entirely on whether a second module ever earns real demand the way the first one needs
to (see `PHASE4_PITCH.md`'s positioning guardrail — this vision statement is explicitly *not*
what gets pitched today).

## Problem

`[HYPOTHESIS — to be corrected by Phase 3]` Labs doing live-imaging developmental biology
currently track cells largely by hand, using general-purpose tools (Fiji/ImageJ + TrackMate,
Icy, CellProfiler, ilastik) that weren't built specifically for 3D time-lapse cell division
tracking, or commercial platforms (Imaris) that are powerful but expensive and still require
substantial manual correction. The actual sharpest pain point — is it time, is it division
detection accuracy, is it cost, is it needing a computational person to run the pipeline — is
not yet known and is exactly what `PHASE3_VALIDATION.md`'s interviews are for.

## Customer & User Personas

`[HYPOTHESIS — all three personas below need Phase 3 correction]`

- **Persona A — the hands-on user.** PhD student or postdoc who personally sits and tracks
  cells. Likely the actual day-to-day user of any tool built here, and probably the highest-value
  Phase 3 interview target.
- **Persona B — the decision-maker.** PI / lab head who doesn't do the manual work but decides
  what tools/licenses the lab adopts and pays for.
- **Persona C — the intermediary.** Core imaging facility staff who support many labs at once —
  potentially a single high-leverage customer (or channel) rather than a single lab.

Which of these is the actual buyer vs. the actual user (they may not be the same person) is a
Business workstream question, not something to guess here.

## Market

Reliable sizing for this specific niche (3D developmental-biology cell tracking software) isn't
readily available — general "life science image analysis software" market reports exist, but
the ones checked while writing this were low-quality report-mill content (templated boilerplate
reused verbatim across unrelated product categories, placeholder values in their own tables),
not something to cite as a real number. One such report claims a "Life Science Image Analysis
Software" market of ~$1.9B (2025) to ~$3.3B (2032, 8.1% CAGR); take that as a rough order of
magnitude at best, not a defensible figure for a pitch deck. The company list it names
(Thermo Fisher, Imaris/Bitplane, Leica Microsystems, Molecular Devices, Zeiss, alongside Fiji as
the dominant free tool) is more trustworthy than its dollar figures, since that same set of
players shows up consistently across independent sources.

`[TO DO]` A defensible market size needs either a paid, reputable report (Frost & Sullivan,
similar) or a bottom-up estimate (number of active developmental-biology labs × realistic
willingness-to-pay from Phase 3) — the latter is probably more honest at this stage than any
top-down report.

## Competitors / Adjacent Tools

Real, existing tools a researcher would compare this against:

| Tool | Type | Notes |
|---|---|---|
| Fiji/ImageJ + TrackMate | Free, open-source | The default in most academic labs; TrackMate does detection + tracking but isn't deep-learning-based by default and division handling requires manual correction |
| Icy | Free, open-source | Similar niche to Fiji, French bioimaging institute-backed |
| CellProfiler | Free, open-source | Broad cell image analysis, strong on segmentation/measurement, less focused on time-lapse tracking specifically |
| ilastik | Free, open-source | Interactive ML-based segmentation/classification, general-purpose |
| Cellpose / StarDist | Free, open-source | Deep-learning segmentation specifically, widely adopted, not a full tracking pipeline on their own |
| Imaris (Oxford Instruments/Bitplane) | Commercial | The incumbent "powerful but expensive" option; already does 3D tracking well, hardest competitor to displace on features |
| btrack (Lowe lab) | Free, open-source | Bayesian tracking library, closer in spirit to this project's tracker |

**Honest read:** most of the pain here likely isn't "no tool exists," it's "the free tools
require manual correction and computational fluency, the paid tool is expensive." That's a
thin-but-real wedge (faster, automated division detection, no license fee) — not "nothing like
this exists," which would be an overclaim. `[TO DO: confirm this framing against Phase 3
interviews — labs may name a completely different frustration.]`

## Value Proposition

`[HYPOTHESIS]` Automated detection, classification, and division-aware tracking through an
upload-and-download workflow, for labs currently spending significant manual time in Fiji/Icy or
paying for Imaris. Sharpened once Phase 3 confirms what actually hurts most.

## Business Model

`[HYPOTHESIS — this is almost entirely open, per the Business workstream]`

Who plausibly pays, and how:
- **Individual labs** — per-analysis or subscription pricing; likely price-sensitive (academic
  grant budgets)
- **Core imaging facilities** — a single facility serving many labs could be a more efficient
  customer than dozens of individual labs; possibly a channel, not just a customer
- **Pharma / biotech** — much higher willingness to pay, different sales motion (enterprise,
  not self-serve), likely needs compliance/support guarantees this project doesn't have yet

Open-core is a plausible model given how much of this space is already open-source (Fiji, Icy,
CellProfiler) — a free/open tracking engine with a paid hosted product (no infra to manage, plus
whatever proprietary refinements exist) might fit the market's existing norms better than a
fully closed product would. `[TO DO: this is a real decision, not a foregone one — see
Legal/IP below, and confirm pricing sensitivity in Phase 3.]`

## Product Roadmap

See the main [README's Roadmap](../README.md#roadmap--future-work) for the technical/product
roadmap in detail. Summary: Phase 1 (core engine) is functionally done except wiring
`LineageBuilder` into the actual inference path; Phase 2 (MVP) has an architecture and an
unverified v0 skeleton; nothing beyond that is built.

## Technical Roadmap

- Wire `src/lineage.py` into `src/predict.py`, `06_pipeline.ipynb`, and
  `kaggle/kaggle_submission.ipynb` (currently a known gap, tracked in the main README)
- Train on more than one frame of one sample (currently a demo-scale model, see the README's
  Results section for why that matters)
- Everything under "Not yet implemented" in the main README's Roadmap

## Scientific Roadmap

See the Science workstream below — detection accuracy, tracking accuracy, runtime, memory
usage, and (where feasible) a head-to-head comparison against TrackMate/Icy on the same data.

`src/evaluate.py` now implements the competition's exact scoring metric (edge Jaccard, adjusted
edge Jaccard, division Jaccard — transcribed from Biohub's published spec, not reconstructed
from memory), with 14 passing tests including one proving it actually accepts
`LineageBuilder`'s real output format. **This closes the "no path to a number" gap, it does not
produce a number.** Nobody has run it against real ground truth yet — that's still the actual
Science workstream deliverable.

## Risks

- **Incumbent risk:** Imaris already does 3D division-aware tracking well and is trusted in the
  field — displacing it on features alone is hard; the wedge is more likely price/openness/ease
  of use than raw capability, at least initially.
- **Generalization risk:** the current model is trained on one dataset (Biohub's Kaggle
  competition data, one imaging modality); performance on a different lab's microscope/staining/
  species is unknown and likely to be worse than the demo numbers suggest.
- **Open-source expectation risk:** academic labs and journals often expect open, reproducible
  methods; a fully closed-source product may face adoption resistance in a way it wouldn't in a
  purely commercial market. Related to the Legal/IP open-source-vs-proprietary question below.
- **Data sensitivity:** uploaded microscopy data may be unpublished research — needs a clear data
  handling/privacy policy before any real lab uploads real data, which doesn't exist yet.
- **Compute cost:** 3D deep learning inference at scale isn't free; the business model needs to
  account for this, not just the license/subscription price.

## Success Metrics (KPIs)

`[HYPOTHESIS — placeholders until the six-week plan produces real numbers]`
- Product: an outside researcher completes the full upload → report workflow with zero help
  from the team
- Science: quantitative detection/tracking accuracy + runtime numbers exist and are compared
  against at least one incumbent tool
- Customer Discovery: 5+ real interviews completed, synthesized, with at least one strong signal
  (per `PHASE3_VALIDATION.md`'s scorecard)
- Business: a stated, reasoned answer (not a guess) to "who pays, and how much"

## Funding Roadmap

`[HYPOTHESIS]` Non-dilutive/incubator support first (matches the current "prove it works, prove
someone wants it" stage); a pre-seed conversation only makes sense once Phase 3 produces real
signal — raising on plans alone, when the plan itself says "these are plans, not evidence,"
would be inconsistent with everything else in this document.
