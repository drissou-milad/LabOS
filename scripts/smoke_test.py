"""
Runs mvp/test_data/synthetic_test.tif through the actual pipeline (mvp/pipeline.py) headlessly
— no browser, no Streamlit — so the whole Upload -> Process -> Track -> Report chain can be
smoke-tested from the command line. See RUNBOOK.md for what this proves and doesn't.

Usage:
    python scripts/smoke_test.py                  # full run, needs torch + a trained checkpoint
    python scripts/smoke_test.py --no-cnn          # skips the CNN filtering step (see below)

--no-cnn runs the detector's raw output straight into tracking/visualization/report, skipping
model loading and classify_centers entirely. This is what let this smoke test actually run in
the sandboxed environment that wrote this repo (no torch there) — it proves everything except
the CNN step works. Run WITHOUT --no-cnn on your own machine for the real, complete test.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tifffile

from mvp.pipeline import run_detection_and_tracking
from src.report import generate_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiff", default=str(PROJECT_ROOT / "mvp" / "test_data" / "synthetic_test.tif"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "mvp" / "test_data" / "smoke_test_report"))
    parser.add_argument("--no-cnn", action="store_true",
                         help="Skip CNN filtering (no torch / no trained checkpoint needed)")
    args = parser.parse_args()

    tiff_path = Path(args.tiff)
    if not tiff_path.exists():
        print(f"{tiff_path} doesn't exist — run `python mvp/make_test_data.py` first.")
        sys.exit(1)

    print(f"Loading {tiff_path}...")
    volume = tifffile.imread(str(tiff_path))
    if volume.ndim == 3:
        volume = volume[:, None, :, :]
    print(f"  shape={volume.shape}")

    print(f"Running detection + tracking (use_cnn_filter={not args.no_cnn})...")
    result = run_detection_and_tracking(volume, use_cnn_filter=not args.no_cnn)
    print(f"  {len(result.nodes)} nodes, {len(result.tracks)} tracks, "
          f"{result.n_divisions()} division(s)")

    out_dir = Path(args.out)
    print(f"Generating report in {out_dir}...")
    paths = generate_report(result, out_dir, volume=volume, sample_name="smoke_test")
    print(f"  report.pdf -> {paths['report_pdf']}")
    print(f"  tracks.csv -> {paths['tracks_csv']}")

    print("\nSMOKE TEST PASSED — the pipeline ran, start to finish, on a real file.")
    if args.no_cnn:
        print("Note: --no-cnn was used, so the CNN filtering step was skipped. Re-run without")
        print("it (needs torch + models/best_model.pth) for the actual complete pipeline.")


if __name__ == "__main__":
    main()
