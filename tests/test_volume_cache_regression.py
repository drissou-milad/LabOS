"""
Regression tests for both reported MVP bugs, driven end-to-end through the REAL app via
streamlit.testing.v1.AppTest (which faithfully reproduces Streamlit's full-script-rerun
semantics -- the exact mechanism bug 1 depends on).

BUG 1 root cause: mvp/streamlit_app.py's _VOLUME_CACHE was a plain module-level dict.
Streamlit re-executes the ENTIRE main script top-to-bottom on every rerun (every widget
interaction), so a plain module-level assignment (`_VOLUME_CACHE = {}`) resets on every
single rerun. The wizard's flow spans at least two reruns between caching the volume (step
2, Dataset) and popping it back out (step 3, Configuration -> "Run Analysis"), so the pop
always returns the fallback `None`, which reaches CellDetector.detect_volume(None), which
does `volume.shape[0]` and crashes with 'NoneType' object has no attribute 'shape'.

BUG 2 root cause: the `datasets` table's schema column (and create_dataset/get_dataset's
actual dict key) is `uploaded_at`, never `created_at` -- streamlit_app.py's Dataset tab
(line ~580) incorrectly reads `dataset['created_at']`, which doesn't exist on that dict, and
raises KeyError.

IMPORTANT IMPLEMENTATION NOTE, worth reading before changing these tests: AppTest.from_file()
execs mvp/streamlit_app.py's source directly into its own fresh module namespace each run --
it does NOT go through a normal `import mvp.streamlit_app`. So monkeypatching attributes on
an `import mvp.streamlit_app as app` object from a fixture has NO effect on what AppTest
actually executes (confirmed empirically while writing this test). Modules streamlit_app.py
itself imports normally (mvp.experiments, mvp.pipeline) ARE sys.modules-cached as usual, so
patching THEIR attributes (see isolated_storage below) does work. Because of this, bug 1's
test cannot intercept process_job with a stub -- it drives the REAL background thread to
completion via status.json polling instead, using a real (untrained -- prediction
correctness is irrelevant here) CellCNN checkpoint so the full pipeline can actually finish.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest

STATUS_POLL_TIMEOUT_S = 30
STATUS_POLL_INTERVAL_S = 0.2


@pytest.fixture
def example_tif():
    """A tiny synthetic (T, Z, Y, X)-shaped TIFF with a few well-separated bright blobs
    (rather than pure noise -- keeps detection counts small and the test fast), dropped into
    the real examples/ directory next to mvp/streamlit_app.py so the app's own
    `PROJECT_ROOT / "examples"` glob finds it. See module docstring for why this can't be
    done via monkeypatching a path instead."""
    examples_dir = PROJECT_ROOT / "examples"
    examples_dir.mkdir(exist_ok=True)
    path = examples_dir / "tiny_regression_case.tif"

    volume = np.random.randint(0, 3, size=(3, 2, 48, 48)).astype(np.uint16)
    for t in range(3):
        for (y, x) in [(20, 20), (24, 28), (28, 16)]:
            volume[t, :, y - 1:y + 2, x - 1:x + 2] = 3000

    tifffile.imwrite(str(path), volume)
    try:
        yield path, volume.shape
    finally:
        path.unlink(missing_ok=True)


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    """Redirects mvp.experiments' and mvp.pipeline's on-disk storage to a temp dir -- see
    module docstring for why patching these (but not mvp.streamlit_app's own attributes)
    actually takes effect under AppTest."""
    import mvp.experiments as experiments
    import mvp.pipeline as pipeline

    storage = tmp_path / "storage"
    storage.mkdir(exist_ok=True)
    monkeypatch.setattr(experiments, "STORAGE_DIR", storage)
    monkeypatch.setattr(experiments, "DB_PATH", storage / "labos_test.db")
    monkeypatch.setattr(pipeline, "STORAGE_DIR", storage)
    return storage


@pytest.fixture
def fake_model_checkpoint(tmp_path, monkeypatch):
    """A real (untrained) CellCNN checkpoint + norm_stats.npz, so run_detection_and_tracking's
    use_cnn_filter=True branch (the default, always taken by the real UI flow) can actually
    load a model and complete -- prediction correctness is irrelevant to either bug; this
    exists only so the pipeline can run to actual completion instead of failing on a missing
    checkpoint for reasons unrelated to what's being tested."""
    import torch
    from src.model import CellCNNNoClassifierReLU
    from src import config

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    checkpoint_path = models_dir / "best_model.pth"
    norm_stats_path = models_dir / "norm_stats.npz"

    torch.save(
    {"model_state_dict": CellCNNNoClassifierReLU().state_dict()},
    checkpoint_path,
     )
    np.savez(norm_stats_path, mean=1500.0, std=300.0)

    monkeypatch.setattr(config, "BEST_MODEL_PATH", checkpoint_path)
    monkeypatch.setattr(config, "BEST_NORM_STATS_PATH", norm_stats_path)
    return checkpoint_path


def _poll_status(storage_dir, experiment_id, timeout=STATUS_POLL_TIMEOUT_S):
    """Polls status.json exactly the way the real app does (mvp/pipeline.py::get_status),
    until it reaches a terminal state ('done' or 'error') or the timeout expires."""
    status_path = storage_dir / experiment_id / "status.json"
    deadline = time.monotonic() + timeout
    last = {"state": "unknown", "message": ""}
    while time.monotonic() < deadline:
        if status_path.exists():
            import json
            last = json.loads(status_path.read_text())
            if last["state"] in ("done", "error"):
                return last
        time.sleep(STATUS_POLL_INTERVAL_S)
    return last


def _run_new_experiment_wizard(at, example_path, experiment_name):
    """Shared driver for both tests: New Experiment -> Experiment Info -> Dataset (example) ->
    Configuration -> Run Analysis. Returns the AppTest instance after landing on the detail
    page (whatever it shows -- callers assert on the outcome)."""
    at.session_state["page"] = "new"
    at.run(timeout=30)

    at.text_input[0].input(experiment_name).run()
    continue_buttons = [b for b in at.button if b.label == "Continue \u2192"]
    assert continue_buttons, "Could not find the step 1 'Continue \u2192' button -- has the wizard's UI changed?"
    continue_buttons[0].click().run()

    example_label = example_path.stem.replace("_", " ").title()
    example_buttons = [b for b in at.button if b.label == example_label]
    assert example_buttons, (
        f"Could not find example dataset button {example_label!r} -- check that "
        f"_render_new_step2_dataset()'s examples_dir logic hasn't changed."
    )
    example_buttons[0].click().run()

    run_buttons = [b for b in at.button if b.label == "Run Analysis"]
    assert run_buttons, "Could not find the 'Run Analysis' button -- has step 3's UI changed?"
    # Generous timeout: landing on the detail page immediately hits the Overview tab's own
    # `while status in (queued, processing): time.sleep(1.5); st.rerun()` loop, which AppTest
    # follows internally until the REAL background process_job thread reaches a terminal
    # state -- the 3s default is nowhere near enough real wall-clock time for that.
    run_buttons[0].click().run(timeout=60)
    return at


def test_volume_survives_new_experiment_wizard(example_tif, isolated_storage, fake_model_checkpoint):
    """BUG 1. Drives the real New Experiment wizard through the exact reported flow and lets
    the real background process_job thread run to completion, polling status.json (exactly
    as the app itself does) rather than trying to intercept the thread. Before the fix: status
    ends as 'error' with a message containing "'NoneType' object has no attribute 'shape'".
    After the fix: status ends as 'done'."""
    example_path, expected_shape = example_tif
    experiment_name = "LabOS End-to-End Validation"

    at = AppTest.from_file(str(PROJECT_ROOT / "mvp" / "streamlit_app.py"))
    at.run(timeout=30)
    at = _run_new_experiment_wizard(at, example_path, experiment_name)

    import mvp.experiments as experiments
    exps = [e for e in experiments.list_experiments() if e["name"] == experiment_name]
    assert len(exps) == 1, f"Expected exactly one matching experiment, found {len(exps)}"
    experiment_id = exps[0]["id"]

    status = _poll_status(isolated_storage, experiment_id)

    assert "nonetype" not in status["message"].lower(), (
        f"THE BUG reproduced: processing failed with a NoneType error: {status['message']!r}. "
        f"This means _VOLUME_CACHE lost the volume across a Streamlit rerun (it must be "
        f"st.session_state-backed, not a plain module-level dict)."
    )
    assert status["state"] == "done", (
        f"Expected status 'done', got {status['state']!r} (message: {status['message']!r}). "
        f"If this is unrelated to the NoneType bug, check fake_model_checkpoint / synthetic "
        f"volume setup, not _VOLUME_CACHE."
    )


def test_dataset_tab_renders_without_keyerror(example_tif, isolated_storage, fake_model_checkpoint):
    """BUG 2. After creating an experiment (same flow as bug 1's test) and landing on its
    detail page, the Dataset tab must render without raising -- before the fix, it raises
    KeyError('created_at') because the datasets table's real column (and get_dataset()'s
    real returned key) is `uploaded_at`, not `created_at`."""
    example_path, expected_shape = example_tif
    experiment_name = "LabOS Dataset Tab Regression"

    at = AppTest.from_file(str(PROJECT_ROOT / "mvp" / "streamlit_app.py"))
    at.run()
    at = _run_new_experiment_wizard(at, example_path, experiment_name)

    # Landing on the detail page (triggered by "Run Analysis" -> _go("detail", ...)) already
    # renders every tab, including Dataset -- no further navigation needed.
    assert not at.exception, (
        f"App raised an exception rendering the experiment detail page: {at.exception}. "
        f"If this mentions 'created_at', that's THE BUG: dataset dicts only ever have "
        f"'uploaded_at' (see mvp/experiments.py's datasets table schema / get_dataset())."
    )
