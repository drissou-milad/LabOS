# Phase 3 — Validation

**The point of this phase is to be wrong cheaply, before Phase 2 gets more expensive.** Five
real conversations with people who currently track cells for a living are worth more than
another week of frontend work — they can invalidate an assumption (upload format, report
format, whether "one video at a time" is even how labs work) that would otherwise only surface
after the MVP is built.

This isn't a survey. It's structured conversation, biased toward specifics over opinions —
"walk me through the last time you did this" beats "would you use a tool like this," because
people are unreliable narrators of their own future behavior and extremely generous with polite
enthusiasm for ideas they will never actually use. (This follows the standard "Mom Test"
approach to customer interviews: ask about their life, not your idea; ask about the past, not
hypotheticals; talk less than they do.)

## Who to talk to

Aim for 5–8 conversations before changing anything, spread across:
- Biology professors running a lab that does live-imaging / developmental biology
- PhD students / postdocs who do the actual hands-on cell tracking
- Core imaging facility staff (they see many labs' workflows, not just one — efficient to talk to)
- Anyone at a company doing this commercially (drug discovery, biotech) if you can reach one

PhD students and postdocs are probably the highest-value first conversations — they're the ones
who actually sit and click through ImageJ/FIJI/Icy at 11pm, not the PI.

## Recruiting message (keep it short, low-pressure)

> Subject: 20 min on how you track cells in microscopy videos?
>
> Hi [name], I'm building a tool for tracking cells through time-lapse microscopy and I'm
> trying to understand how people currently do this before building more of it. Would you have
> 20 minutes this week or next to walk me through your current process? I'm not selling
> anything yet — genuinely trying to learn.

Don't attach a demo or describe the product here. The goal is to hear about their current
workflow uncontaminated by your idea.

## Interview guide

### 1. Open (2 min)

State plainly this is research, not a sales call, and you want to hear about problems, not
reactions to a pitch. Ask permission to take notes.

### 2. Current workflow — the core of the interview (10–12 min)

Anchor everything to a **specific, recent, real instance**, not "generally what do you do":

- "Tell me about the last time you needed to track cells across a time-lapse video. Walk me
  through it start to finish."
- What software did you use? (ImageJ/FIJI, Icy, TrackMate, Imaris, custom scripts, something
  else?)
- What file format did the raw data come in? *(directly resolves the Phase 2 upload-format
  question — don't skip this one)*
- How long did that take, start to finish? Was that mostly waiting, or mostly hands-on work?
- Who does this in your lab — you, a specific person, everyone?
- What was the most annoying or time-consuming part of it?
- Did the software handle cell divisions / lineage tracking, or did you do that by hand?
- What did you do with the results afterward? Who looked at them, in what format? *(resolves
  the "what does a useful report actually look like" question)*
- Is this a one-off analysis, or something you do repeatedly / would want to batch across many
  videos? *(resolves the single-video-vs-batch question)*

### 3. Quantify the cost (3–5 min)

- Roughly how often does this come up — weekly, per-project, rarely?
- If you could get this same result in 10 minutes instead of [however long they said], would
  that change what you're willing to analyze? (Are they currently *not* analyzing things
  because it's too slow/tedious — that's a stronger signal than "existing workflow is annoying.")

### 4. React to the concept — only now (5 min)

Describe the workflow briefly (upload → AI processing → interactive tracking → download
report), or show a screenshot/screen-share if you have one. Then go quiet and listen for what
they bring up unprompted, rather than asking leading questions.

Watch for the difference between:
- **Weak signal:** "That's cool," "I could see us using that," generic enthusiasm with no
  specifics attached.
- **Strong signal:** unprompted questions about pricing, asking whether it handles their
  specific microscope's file format, asking if they can try it on their own data now, asking
  who else is using it, offering to introduce you to a colleague or their PI.

### 5. Close with a real ask (2 min)

Not "would you use this" — get an actual commitment:

- "Can I send you a version to try on one of your own videos in the next two weeks?"
- "Would you be open to a follow-up once there's something to test?"

A "yes, send it" that they later ignore is itself useful data (weak signal). A "yes" with no
follow-through requested is close to no signal at all.

## Signal scorecard (fill in after each interview)

| | Weak signal | Strong signal |
|---|---|---|
| Reaction | Polite, generic, opinion-based | Specific, asks logistics questions, self-initiated |
| Commitment | "Sounds interesting" | Agrees to try it on real data, offers an intro |
| Current pain | Mildly annoying, tolerable | Actively avoided doing some analyses because of it |
| Specificity | Vague answers about their workflow | Names exact tools, exact file formats, exact time spent |

## Synthesis (after 5–8 interviews)

Fill in one row per interview — patterns matter more than any single conversation:

| Person / role | Current tool(s) | File format | Time spent per analysis | Handles divisions? | Biggest pain point | Signal (weak/medium/strong) |
|---|---|---|---|---|---|---|
| | | | | | | |

Look specifically for:
- Do 3+ people mention the same file format? → answers the Phase 2 upload question with
  evidence instead of a guess.
- Do 3+ people mention the same pain point? → that's your actual value proposition, possibly
  different from what you assumed it was.
- Is "strong signal" clustered in one role (e.g. postdocs, not PIs)? → tells you who the actual
  buyer/user is, which changes how Phase 4 framing should read.

## What "validated" actually means here

Not "people said nice things." It means: real file formats are known (not guessed), the
sharpest pain point is known (not assumed), and at least a couple of people have agreed to
actually try the tool on real data. That's the bar before Phase 4 materials should claim any
validation at all — see the guardrail at the top of `PHASE4_PITCH.md`.
