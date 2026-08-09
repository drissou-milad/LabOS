"""
Measures actual memory usage of the detection -> tracking -> report pipeline, stage by stage.

Usage:
    python scripts/profile_memory.py                          # uses the shipped synthetic file
    python scripts/profile_memory.py --tiff path/to/real.tif   # use your own (larger) file
    python scripts/profile_memory.py --no-cnn                  # skip the CNN step (no torch needed)

Uses psutil (in requirements.txt) for RSS (resident set size — the actual physical memory the
process is using), which is a more honest number for "will this fit in RAM" questions than
Python-level allocation tracking would be, since it includes NumPy/PyTorch's own buffers.
"""

import argparse
import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psutil
import tifffile

from mvp.pipeline import run_detection_and_tracking
from src.report import generate_report

_process = psutil.Process()


def _rss_mb():
    return _process.memory_info().rss / (1024 * 1024)


def _measure(label, fn, baseline):
    gc.collect()
    before = _rss_mb()
    result = fn()
    gc.collect()
    after = _rss_mb()
    print(f"{label:40s}  RSS: {after:8.1f} MB  (+{after - before:7.1f} MB this stage, "
          f"+{after - baseline:7.1f} MB total)")
    return result, after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiff", default=str(PROJECT_ROOT / "mvp" / "test_data" / "synthetic_test.tif"))
    parser.add_argument("--no-cnn", action="store_true")
    args = parser.parse_args()

    tiff_path = Path(args.tiff)
    if not tiff_path.exists():
        print(f"{tiff_path} doesn't exist. Run `python mvp/make_test_data.py` first, or pass --tiff.")
        sys.exit(1)

    baseline = _rss_mb()
    print(f"Baseline RSS (Python + imports, before loading data): {baseline:.1f} MB")
    print(f"File: {tiff_path}\n")

    volume, baseline_after_load = _measure(
        "Load volume (tifffile.imread)",
        lambda: tifffile.imread(str(tiff_path)),
        baseline,
    )
    if volume.ndim == 3:
        volume = volume[:, None, :, :]
    print(f"  shape={volume.shape} dtype={volume.dtype} "
          f"array size={volume.nbytes / (1024*1024):.2f} MB\n")

    result, after_pipeline = _measure(
        f"Detection + tracking (use_cnn_filter={not args.no_cnn})",
        lambda: run_detection_and_tracking(volume, use_cnn_filter=not args.no_cnn),
        baseline,
    )
    print(f"  {len(result.nodes)} nodes, {len(result.tracks)} tracks, "
          f"{result.n_divisions()} division(s)\n")

    out_dir = PROJECT_ROOT / "mvp" / "test_data" / "memory_profile_report"
    _, after_report = _measure(
        "Generate report (visualize + evaluate wiring + PDF)",
        lambda: generate_report(result, out_dir, volume=volume, sample_name="memory_profile"),
        baseline,
    )

    peak = _process.memory_info().rss / (1024 * 1024)
    print(f"\nPeak RSS observed: {peak:.1f} MB")
    print(f"Volume array itself: {volume.nbytes / (1024*1024):.2f} MB "
          f"({100 * volume.nbytes / (1024*1024) / peak:.1f}% of peak RSS)")

    print("\n--- Caveat ---")
    print(f"This ran on a {volume.shape} synthetic volume ({volume.nbytes/(1024*1024):.1f} MB).")
    print("Real microscopy volumes are typically much larger (multi-GB). Memory scales roughly")
    print("with volume size for the detection/tracking stages (they hold the full volume in")
    print("memory at once — see src/detector.py's detect_volume() and mvp/pipeline.py's")
    print("run_detection_and_tracking()), so extrapolate cautiously, and re-run this script")
    print("directly against a real, large file before trusting a memory estimate for one.")


if __name__ == "__main__":
    main()
