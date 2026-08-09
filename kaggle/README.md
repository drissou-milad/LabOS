# Kaggle Submission Workflow

This competition doesn't accept a raw CSV upload — Kaggle needs to execute a **Kaggle
Notebook** itself, and the `submission.csv` it produces is what gets scored. That notebook runs
with **internet access off**, so `src/` and the trained model can't be cloned/`pip install -e .`'d
from GitHub at submission time. They need to already exist inside Kaggle as a **Dataset**,
attached to the notebook as an input.

## 1. Package `src/` + `models/` as a Kaggle Dataset

From the repo root, with the Kaggle CLI configured (`kaggle.json` in `~/.kaggle/`):

```bash
mkdir -p kaggle_dataset_upload
cp -r src kaggle_dataset_upload/src
cp -r models kaggle_dataset_upload/models

cd kaggle_dataset_upload
kaggle datasets init -p .
```

This creates `dataset-metadata.json`. Edit it — Kaggle needs a unique `id` in the form
`<your-kaggle-username>/<dataset-slug>`:

```json
{
  "title": "biohub-cell-tracking-src",
  "id": "<your-kaggle-username>/biohub-cell-tracking-src",
  "licenses": [{"name": "MIT"}]
}
```

Then create it:

```bash
kaggle datasets create -p .
```

**Whenever you retrain the model or change `src/`, re-upload with:**

```bash
kaggle datasets version -p . -m "describe what changed"
```

`kaggle_submission.ipynb`'s `CODE_DATASET_SLUG` variable must match the `id` you chose above.

## 2. Create the Kaggle Notebook

1. On the competition page → **Code** → **New Notebook**.
2. Copy the contents of `kaggle/kaggle_submission.ipynb` in (or upload the file directly —
   Kaggle notebooks accept `.ipynb` upload).
3. **Add Input** (right sidebar):
   - the competition's own data (search for this competition under "Competitions")
   - the dataset you just created (search for its title under "Your Datasets")
4. **Turn internet OFF** (right sidebar, "Internet" toggle) — required for the Submit button to
   be available on most competitions, and this notebook doesn't need it once the dataset above
   is attached.
5. Edit `CODE_DATASET_SLUG` near the top of the notebook to match your dataset's `id`.

## 3. Run it

**Save Version → Save & Run All.** Kaggle executes the whole notebook on its servers. If it
finishes without error and `submission.csv` is in that version's output, the **Submit** button
becomes available on that version — click it to actually enter the competition.

## Before you trust the output: verify the submission schema

`kaggle_submission.ipynb` maps its internal detections/tracks into a guessed
`(id, t, z, y, x, parent_id)`-style CSV — a reasonable shape for a tracking-graph competition,
but **not one I could confirm against this competition's actual `sample_submission.csv`**
(Kaggle's Data page is JavaScript-rendered and unreadable from outside Kaggle). The notebook's
cell 7 prints the real `sample_submission.csv` columns specifically so you can check this before
submitting — don't skip it.

## Known limitation

`kaggle_submission.ipynb` currently does frame-to-frame tracking only, via `HungarianTracker`
directly — no cell-division/lineage prediction reaches `submission.csv` yet. Every submitted
node has at most one parent, so the division-related portion of the competition score will be
zero.

This is a wiring gap, not a missing capability: `src/lineage.py` (`LineageBuilder`) already
detects divisions and assigns CTC-style track IDs on top of `HungarianTracker`, with its own
test suite (`tests/test_lineage.py`). It just isn't plugged into this notebook, `src/predict.py`,
or `06_pipeline.ipynb` yet — see the main [README's Roadmap](../README.md#roadmap--future-work).
