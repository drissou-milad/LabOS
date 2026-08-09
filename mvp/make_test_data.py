"""
Generates a small synthetic multi-frame TIFF stack with a few Gaussian-blob "cells" that drift
and one that divides — enough to smoke-test the whole Upload -> Process -> Track -> Report
pipeline without the real (large, Kaggle-auth-gated) competition dataset, and without a GPU.

This is a synthetic sanity check, not a source of real accuracy numbers — same caveat as
src/benchmark.py's synthetic demo table. Its only job is proving the pipeline runs.

Usage:
    python mvp/make_test_data.py
    # writes mvp/test_data/synthetic_test.tif

Then either:
    - Upload mvp/test_data/synthetic_test.tif directly into the running Streamlit app, or
    - Run scripts/smoke_test.py, which does the same thing without a browser at all (see
      RUNBOOK.md).
"""

from pathlib import Path

import numpy as np
import tifffile


def _blob(shape, center, sigma=3.0, amplitude=2500):
    """amplitude=2500, not something smaller and more 'realistic'-looking, because
    CellDetector Gaussian-smooths (sigma=config.GAUSSIAN_SIGMA=2) before thresholding at
    config.DETECTION_THRESHOLD=800. Blurring a sigma=3 Gaussian with a sigma=2 Gaussian
    attenuates its peak to roughly amplitude * 3^2/(3^2+2^2) = amplitude * 9/13 — so an
    amplitude of 800 (this file's first version) peaked at ~554 after smoothing, comfortably
    *below* the detection threshold, and every single detection silently vanished. Caught by
    actually running the detector on this file and getting 0 nodes back, not by reasoning
    about it in the abstract — see RUNBOOK.md.
    """
    yy, xx = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    return amplitude * np.exp(-(((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2 * sigma ** 2)))


def make_synthetic_volume(n_frames=6, shape=(96, 96), n_z=3, seed=0, flavor="default"):
    """
    Returns a (T, Z, Y, X) uint16 array. `flavor` controls how many cells and how much
    division activity, so the three gallery categories (examples/) are actually
    distinguishable when run through the real pipeline, not just differently named:

    - "sparse": 2 cells, no divisions, well separated
    - "crowded": 10 cells, no divisions, densely packed
    - "division_rich": 3 cells, each dividing at least once
    - "default": the original 2-cells-one-division behavior (used by the smoke test)
    """
    rng = np.random.default_rng(seed)
    volume = np.zeros((n_frames, n_z, *shape), dtype=np.float32)

    if flavor == "sparse":
        n_cells, division_frames = 2, []
    elif flavor == "crowded":
        n_cells, division_frames = 10, []
    elif flavor == "division_rich":
        n_cells, division_frames = 3, [n_frames // 3, (2 * n_frames) // 3]
    else:
        n_cells, division_frames = 2, [n_frames // 2]

    margin = 15
    min_initial_spacing = 14  # just above config.CELL_RADIUS (12px) so cells start resolvable
    positions = []
    attempts = 0
    while len(positions) < n_cells and attempts < n_cells * 200:
        candidate = np.array([
            rng.uniform(margin, shape[0] - margin), rng.uniform(margin, shape[1] - margin)
        ])
        if all(np.linalg.norm(candidate - p) >= min_initial_spacing for p in positions):
            positions.append(candidate)
        attempts += 1
    if len(positions) < n_cells:
        # Couldn't fit all of them with the spacing constraint in this canvas size — proceed
        # with however many fit rather than silently produce fewer cells than requested without
        # saying so.
        print(f"Warning: only placed {len(positions)}/{n_cells} cells with "
              f"{min_initial_spacing}px minimum spacing in a {shape} canvas.")
    velocities = [rng.uniform(-0.8, 0.8, 2) for _ in range(len(positions))]
    extra_cells = []  # (position, velocity, birth_frame) for daughters born during the run

    def clip_to_canvas(pos):
        return np.array([np.clip(pos[0], margin, shape[0] - margin),
                          np.clip(pos[1], margin, shape[1] - margin)])

    for t in range(n_frames):
        frame = np.zeros(shape, dtype=np.float32)
        for pos in positions:
            frame += _blob(shape, pos)
        for pos, _vel, _birth in extra_cells:
            frame += _blob(shape, pos)

        frame += rng.normal(0, 20, shape).clip(min=0)
        for z in range(n_z):
            defocus = 1.0 - 0.15 * abs(z - n_z // 2)
            volume[t, z] = frame * defocus

        for i in range(len(positions)):
            positions[i] = clip_to_canvas(positions[i] + velocities[i] + rng.normal(0, 0.3, 2))
        extra_cells = [
            (clip_to_canvas(pos + vel + rng.normal(0, 0.3, 2)), vel, birth)
            for pos, vel, birth in extra_cells
        ]

        if t in division_frames and positions:
            parent_idx = t % len(positions)  # cycle through which cell divides, for variety
            parent_pos = positions[parent_idx]
            daughter_pos = parent_pos.copy() + np.array([14.0, -14.0])
            daughter_vel = velocities[parent_idx] + np.array([1.0, -1.0])
            extra_cells.append((daughter_pos, daughter_vel, t))

    return volume.clip(0, 65535).astype(np.uint16)


def main():
    out_dir = Path(__file__).resolve().parent / "test_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_test.tif"

    volume = make_synthetic_volume()
    tifffile.imwrite(str(out_path), volume)

    print(f"Wrote {out_path}  shape={volume.shape}  dtype={volume.dtype}")
    print("Upload this file directly into the running Streamlit app, or run scripts/smoke_test.py.")


if __name__ == "__main__":
    main()
