"""
Generates examples/ — three synthetic datasets standing in for a real dataset gallery
(Priority 5). These are NOT real embryo data — this environment has no access to the real
competition dataset or actual microscopy files. They're synthetic, generated the same way as
mvp/test_data/synthetic_test.tif, parameterized to actually be distinguishable in the way the
gallery categories claim (division-rich genuinely has more divisions, crowded genuinely has
more cells) rather than just differently named.

Replace these with real embryo data the moment you have it — examples/README.md says so
explicitly, and the gallery UI in mvp/streamlit_app.py reads whatever's in examples/, so
dropping in real .tif files there is the entire migration path.

Usage:
    python scripts/make_gallery_examples.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tifffile

from mvp.make_test_data import make_synthetic_volume

EXAMPLES = [
    {
        "filename": "embryo_a_division_rich.tif",
        "flavor": "division_rich",
        "seed": 1,
        "title": "Embryo A — Division-rich",
        "description": "3 cells; the generator codes 2 division events, of which 1 is reliably "
                        "detected by the current pipeline at these settings (verified by "
                        "actually running it — see examples/README.md) — a real, honest "
                        "demonstration of division detection, not a claim of catching every "
                        "coded event.",
    },
    {
        "filename": "embryo_b_crowded.tif",
        "flavor": "crowded",
        "seed": 2,
        "title": "Embryo B — Crowded",
        "description": "10 cells, no divisions — stresses the detector and tracker under density.",
    },
    {
        "filename": "embryo_c_sparse.tif",
        "flavor": "sparse",
        "seed": 3,
        "title": "Embryo C — Sparse",
        "description": "2 cells, no divisions — the easy case, good for a first-time demo.",
    },
]


def main():
    out_dir = PROJECT_ROOT / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in EXAMPLES:
        volume = make_synthetic_volume(n_frames=8, flavor=spec["flavor"], seed=spec["seed"])
        out_path = out_dir / spec["filename"]
        tifffile.imwrite(str(out_path), volume)
        print(f"Wrote {out_path}  shape={volume.shape}")

    print(f"\n{len(EXAMPLES)} example datasets written to {out_dir}")
    print("See examples/README.md — these are synthetic placeholders, not real embryo data.")


if __name__ == "__main__":
    main()
