# Incubator Demo Script — 5 Minutes

Practice this out loud, with a timer, at least 3 times before the real thing. The goal is that
you never have to think about what to click next — all your attention goes to what you're
saying, not the UI.

## Before you start (do this the night before, not 10 minutes before)

- [ ] Restart your machine's Streamlit process fresh — `streamlit run mvp/streamlit_app.py` —
      and confirm it opens cleanly, no leftover error state from a previous session.
- [ ] Delete or archive any test/junk experiments from "My Experiments" so the Dashboard looks
      like a real workspace, not a debugging log.
- [ ] Pick ONE dataset for the demo and use the exact same one every rehearsal — muscle memory
      matters more than variety here. A real public dataset (see `RUNBOOK.md`'s CTC step) is a
      stronger opener than a synthetic example, if you have time to run it in advance.
- [ ] Pre-create one OTHER experiment that's already `done`, so if live analysis is slow or
      flaky on the day, you have a populated Experiment Detail / Reports / Performance page as
      a fallback without needing the live run to finish.
- [ ] Have `docs/EVIDENCE_PACKAGE_TrackMate_Comparison.md`'s numbers memorized well enough to
      say one sentence from memory if asked, without needing to open the file.
- [ ] Close every other browser tab. Full screen. Zoom level checked on the actual presentation
      display, not just your laptop.

## The script (target: 5:00, budget below leaves ~30s slack)

**[0:00–0:30] Open on the Dashboard.**
> "This is LabOS — a research operating system for cell tracking and lineage analysis. What
> you're looking at is real activity from experiments we've actually run, not a mockup."
Point at Recent Activity / real stats briefly. Don't linger.

**[0:30–1:00] New Experiment.**
Click "+ New Experiment." Walk through the 3-step wizard fast:
> "Every experiment starts the same way — name it, attach a dataset, pick or create an analysis
> configuration."
Type a name. Don't overthink it, don't backspace and retype — whatever you type is fine.

**[1:00–1:30] Upload microscopy data.**
Select your pre-chosen dataset (upload or example button — whichever is faster and more
reliable on the day).
> "This is real [public / your lab's] microscopy data — [one sentence on what it is: cell type,
> what's being imaged]."

**[1:30–2:00] Configure analysis.**
Show the Configuration step. If you have a saved Configuration, pick it and say why:
> "This Configuration is tuned for [this cell type / this imaging modality] — parameters here
> aren't guesses, they're saved and reusable across experiments."
Click Run Analysis.

**[2:00–2:30] While it runs, talk, don't wait in silence.**
> "While that's processing — detection, tracking, and lineage reconstruction all happen in one
> pipeline, so let me switch to an experiment I ran earlier so you can see the full output."
Switch to your pre-done fallback experiment. This is the safety net — don't wait on the live
run if it's taking more than ~20–30 seconds.

**[2:30–3:15] Viewer.**
> "This is the interactive viewer — every detected cell, tracked across time, colored by
> lineage." Move the frame slider once or twice. Hover a cell if you have time.

**[3:15–3:45] Lineage.**
> "This is the actual lineage tree — every division event, reconstructed automatically."
Use the highlight dropdown on one track if there's a division to show off.

**[3:45–4:15] Generate / open PDF report.**
> "One click produces a publication-formatted report — this isn't a screenshot, it's a real
> PDF with the tracking data, lineage tree, and summary statistics." Open it, scroll once.

**[4:15–4:40] Export.**
> "And the underlying data exports in standard formats — CSV, and Cell Tracking Challenge
> format for anyone who wants to compare against other tools." Show the download buttons, don't
> actually wait for a download to finish on stage.

**[4:40–5:00] Close on validation, not features.**
> "We've validated this against [N] real public datasets and TrackMate, the standard tool in
> this space — [one real number from your Performance page or evidence package]. And we've
> talked to [N] researchers directly about what actually matters to them day-to-day."
Stop talking. Let the last sentence land. Don't keep clicking.

## If something breaks live

- Don't apologize more than once. One "let me switch to a backup" and move on.
- This is exactly why you have the pre-done fallback experiment from step 0 — use it
  immediately rather than debugging live.
- If the whole app is down: you still have `docs/EVIDENCE_PACKAGE_TrackMate_Comparison.md` and
  a PDF report saved locally — walk through those instead. Keep a PDF report and one screenshot
  of the Lineage view saved outside the app (email attachment / phone) as an absolute last
  resort.

## After rehearsal 1, time yourself honestly

If you're over 5:30, cut from Viewer or Export first — Configuration, Lineage, and the closing
validation line are the parts that differentiate LabOS from "yet another tracking script."
