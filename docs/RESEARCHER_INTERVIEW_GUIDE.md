# Researcher Interview Guide — v0.7.0 Validation

**Goal:** 5 conversations, 20–30 minutes each, in person or video call. You're not selling
LabOS here — you're trying to find out if it solves a real problem, and you want to hear "no"
as clearly as "yes." An incubator will trust five honest, specific answers more than fifty
polite ones.

## Who to talk to

Aim for variety, not five of the same role — the product review specifically flagged this:
- 1 biology professor / PI (owns budget & priorities, thinks in publications)
- 1–2 PhD students (do the actual day-to-day imaging work, most time-pressured)
- 1 lab technician (runs the microscope, may not do the analysis themselves)
- 1 microscopy core-facility researcher (sees many labs' workflows, good pattern-matcher)

If you only have access to one or two of these types, do 5 conversations anyway rather than
skipping the milestone — variety helps, but 5 real conversations beats 0 perfectly-sourced ones.

## Before each conversation

- Don't demo LabOS first. Ask about their current workflow first, unprompted — if you show the
  product before asking what's frustrating, you'll get reactions to your product instead of an
  honest account of their problem.
- Have a notebook or a doc open per person. Write down close-to-verbatim quotes where you can —
  "it takes forever" is weaker evidence than "I spend about 6 hours a week manually re-tracking
  cells TrackMate lost after a division."

## Questions (roughly in this order)

**Warm-up / current workflow (5 min)**
1. Walk me through what you do today, from raw microscopy data to a result you'd put in a
   paper or share with your PI.
2. What tool(s) do you currently use for cell tracking / lineage analysis?

**The frustration question — spend the most time here (10 min)**
3. What's the most frustrating or time-consuming part of that process?
4. Follow up hard on whatever they say — ask "why is that frustrating specifically?" and "how
   often does that happen?" until you get something concrete (a number, a specific failure
   mode), not just a vibe.
5. Has that ever caused you to redo work, miss a deadline, or leave something out of a paper?

**Only after 3–5: show LabOS (5–10 min)**
6. Walk them through the real 5-minute demo (see `docs/DEMO_SCRIPT.md`).
7. Would this save you time? Where specifically, and roughly how much?
8. Which single feature matters most to you — Viewer, Lineage, Reports/PDF export, the
   TrackMate-comparable Configuration control, something else?
9. What's missing that would stop you from actually using this?

**Wrap-up (2 min)**
10. Would you be willing to try this on one of your own real datasets? (This is your best
    lead for turning an interview into an actual pilot user — always ask.)
11. Is there anyone else you'd recommend I talk to?

## After each conversation

Fill in one row of `docs/RESEARCHER_FEEDBACK.md` (template below) within the same day, while
it's fresh — don't batch this at the end of all 5.

## What NOT to do

- Don't argue if they say something isn't valuable. Write it down. A defensive reaction here
  costs you the honest data this milestone exists to produce.
- Don't lead the witness — "wouldn't it be great if LabOS could do X" gets you an agreeable
  nod, not evidence.
- Don't skip the negative answers when you write this up for the incubator. "3 of 5 said
  Reports mattered most; 2 said Viewer" is more credible than a page of only positive quotes.
