# Evidence Package — LabOS vs TrackMate

**Rule for filling this in:** if TrackMate wins a row, write that TrackMate wins. This document
is only worth showing an incubator if every number in it is one you'd defend under questioning.

## 1. Setup

| | |
|---|---|
| Date run | |
| Machine (OS, CPU/RAM, GPU if any) | |
| LabOS version | v0.5.1 |
| Fiji / TrackMate version | |
| Dataset | (name + source URL/DOI — must be real/public, not `examples/`'s synthetic gallery) |
| Dataset size | (frames × resolution, or file size) |
| Ground truth source | (competition annotations / manual annotation / other) |

## 2. Runtime

| Method | Wall-clock time | What's included |
|---|---|---|
| LabOS | ___ s (from `scripts/run_trackmate_comparison.py`'s printed table) | detection + tracking + lineage, this run only |
| TrackMate | ___ s (stopwatch, click-to-done in Fiji) | detection + tracking, however you configured it |

Note anything that makes this comparison less apples-to-apples (e.g. LabOS ran headless on a
loaded machine, TrackMate ran interactively with a GUI open, one used GPU and the other didn't).

## 3. Accuracy

Fill in from `scripts/run_trackmate_comparison.py`'s printed table (both rows scored against
the identical ground truth, identical `max_distance` tolerance):

| Metric | LabOS | TrackMate |
|---|---|---|
| Node detection rate | | |
| Edge precision | | |
| Edge recall | | |
| Adjusted edge Jaccard | | |
| Division Jaccard | | |
| Combined tracking score | | |

## 4. Division detection (qualitative, beyond the Division Jaccard number above)

- Number of true divisions in this dataset: ___
- LabOS: correctly found ___, missed ___, false-positive ___
- TrackMate: correctly found ___, missed ___, false-positive ___
- Anything either method got structurally wrong (e.g. merged two daughters back into one track,
  split a non-dividing cell)?

## 5. Manual intervention required

Be specific and honest — this is usually where the real product story lives, not the accuracy
numbers.

**LabOS:**
- [ ] Uploaded file, picked a Configuration, clicked Run — nothing else
- [ ] Had to create/tune a Configuration first (list which parameters, and why)
- [ ] Had to manually correct any tracks after the run
- Total hands-on time: ___

**TrackMate:**
- [ ] Detector settings tuned (list which, and how many attempts)
- [ ] Tracker settings tuned (list which)
- [ ] Gap-closing / merging / splitting settings configured
- [ ] Manual track editing/correction in the GUI (how much?)
- Total hands-on time: ___

## 6. Report generation

**LabOS:** (does the Reports tab produce something usable as-is? attach the PDF)

**TrackMate:** (what did you have to do to get anything shareable — CSV export, screenshot,
manual write-up?)

## 7. Honest summary

Two or three sentences. Where LabOS is ahead, where TrackMate is ahead, and — critically — for
which kind of dataset/use case each result actually holds (a result on one small synthetic-ish
public dataset does not generalize to "LabOS is better than TrackMate," and shouldn't be
written as if it does).

## 8. Raw output

Paste `scripts/run_trackmate_comparison.py`'s full printed output here as an appendix, so
anyone can check the summary above against the actual numbers.

```
(paste here)
```
