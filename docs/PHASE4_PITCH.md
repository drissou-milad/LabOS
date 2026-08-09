# Phase 4 — Incubator Version

## Guardrail — read this before writing a single slide

**Nothing in this document should claim validation evidence that doesn't exist yet.**
Incubators and investors have seen thousands of decks; the fastest way to lose credibility with
an experienced reviewer is a "traction" slide that doesn't survive one follow-up question ("how
many of those X interviews actually agreed to try it?"). If Phase 3 hasn't happened yet, the
honest version of this deck says so — a team that clearly knows what stage it's at reads as
more credible, not less, than one overclaiming.

Every placeholder below marked `[FROM PHASE 3]` should stay a placeholder, with real numbers
from `PHASE3_VALIDATION.md`'s synthesis table, until it's actually true.

## What company are we building? — the actual decision

The instinct to frame this as "AI Platform for Biological Image Analysis" rather than "Cell
Tracking Software" is directionally right — platforms are more valuable than point tools, and
segmentation/counting/colony analysis are natural extensions of the same underlying
detection-and-tracking engine. **The timing is the issue, not the ambition.**

Recommendation: **lead with the wedge, hold the platform framing in reserve.**

- **Now (pre- or early-Phase 3):** "We built a tool that tracks cells through 3D microscopy and
  automatically catches cell divisions — something labs currently do by hand in ImageJ." Narrow,
  concrete, verifiable. Nobody can poke a hole in it because it's just true.
- **Later (once a second module has its own evidence of demand, not just cell tracking):** "We
  started with cell tracking, and now researchers are asking us for segmentation / colony
  analysis / X" — this is a *much* stronger platform pitch than the same claim made speculatively,
  because it's backed by pull from actual users rather than your own roadmap imagination.

This isn't "never mention the platform vision" — it's "mention it as where this is headed, in
one sentence, not as the headline you lead with." A reviewer who asks "so what's actually built
and used today?" should get an answer that matches the wedge pitch, not the platform pitch.

## One-liner options

Use the narrow one until Phase 3 evidence exists; graduate to the broader one once it does.

- **Now:** "LabOS automatically detects, classifies, and tracks cells — including
  divisions — through 3D microscopy videos, for labs currently doing this by hand."
- **Later, once earned:** "We're building the AI layer for biological image analysis, starting
  with cell tracking — labs are already asking us for [segmentation/counting/whatever Phase 3
  or early users actually asked for]."

## Pitch deck outline

Structure, not full slide content — fill each section from what's actually true.

**1. Problem**
Labs tracking cells through time-lapse microscopy today do it largely by hand (ImageJ/FIJI,
Icy, custom scripts), which is slow and doesn't reliably catch cell divisions.
`[FROM PHASE 3: exact time-per-analysis numbers, exact tool names, from the synthesis table]`

**2. Solution**
One-line description of the pipeline (detect → classify → track → detect divisions → report),
plus the "upload → processing → interactive tracking → download" workflow from Phase 2.

**3. Demo**
Screenshots or a short recording of the actual v0 MVP (`mvp/streamlit_app.py`) — real output,
not mockups. The architecture and division-detection diagrams already in this repo's `docs/`
and main README are reusable here directly.

**4. Why now / why us**
Whatever's true — e.g. cheaper compute for 3D deep learning, a public annotated dataset
(Biohub's Kaggle competition data) existing to train on, founder background, etc. Don't pad this
with generic AI-hype language; specific reasons read better than vague ones.

**5. Market**
Standard TAM/SAM/SOM only if you can back the numbers with a real source (number of active
developmental-biology labs, imaging facility counts, etc.) — a market slide with unsourced
numbers is worse than no market slide.

**6. Validation / Traction**
`[FROM PHASE 3, ONLY]`: number of researchers interviewed, the specific pain points that came
up repeatedly, how many agreed to try the tool on real data, any actual usage if the MVP has
been tried by a real lab yet. If this section would currently be empty, say "Phase 3 in
progress, N interviews completed" rather than skip the slide or fill it with hope.

**7. Roadmap (platform vision goes here, not on slide 1)**
Cell tracking now → segmentation / counting / colony analysis / automated reporting as
extensions of the same core engine, framed as "next" not "already validated."

**8. Team**

**9. Ask**
What you actually want from this incubator/investor — money, intros to labs, technical
mentorship, credibility. Be specific; "help us grow" isn't an ask.

## Before submitting anywhere: honesty checklist

- [ ] Every traction/validation number on the traction slide traces back to a real row in
  `PHASE3_VALIDATION.md`'s synthesis table
- [ ] The demo shown is the actual `mvp/streamlit_app.py` output (or a fixed version of it),
  not a mockup presented as if it's the working product
- [ ] The platform-vision slide is clearly future framing, not stated as already true
- [ ] You can answer "what's actually built and working today?" with the same answer you'd give
  a skeptical engineer, not a more impressive-sounding one
