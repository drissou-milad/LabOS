import numpy as np

from mvp.make_test_data import make_synthetic_volume
from mvp.pipeline import run_detection_and_tracking


GALLERY_DETECTOR_CONFIG = {
    "detection_threshold": 20,
}


def test_flavor_generation_is_reproducible():
    v1 = make_synthetic_volume(n_frames=6, flavor="division_rich", seed=1)
    v2 = make_synthetic_volume(n_frames=6, flavor="division_rich", seed=1)
    assert np.array_equal(v1, v2)


def test_different_seeds_produce_different_volumes():
    v1 = make_synthetic_volume(n_frames=6, flavor="sparse", seed=1)
    v2 = make_synthetic_volume(n_frames=6, flavor="sparse", seed=2)
    assert not np.array_equal(v1, v2)


def test_crowded_flavor_has_more_cells_than_sparse():
    """The actual point: the gallery categories need to be genuinely distinguishable through
    the real pipeline, not just differently named. Checked here so it can't silently drift
    out of true as the generator or detector settings change."""
    crowded = make_synthetic_volume(n_frames=4, flavor="crowded", seed=2)
    sparse = make_synthetic_volume(n_frames=4, flavor="sparse", seed=3)

    crowded_result = run_detection_and_tracking(
        crowded,
        use_cnn_filter=False,
        config_params=GALLERY_DETECTOR_CONFIG,
    )
    sparse_result = run_detection_and_tracking(
        sparse,
        use_cnn_filter=False,
        config_params=GALLERY_DETECTOR_CONFIG,
    )

    crowded_frame0 = sum(1 for n in crowded_result.nodes if n.t == 0)
    sparse_frame0 = sum(1 for n in sparse_result.nodes if n.t == 0)
    assert crowded_frame0 > sparse_frame0


def test_division_rich_flavor_produces_at_least_one_division():
    volume = make_synthetic_volume(
        n_frames=8,
        flavor="division_rich",
        seed=1,
    )

    result = run_detection_and_tracking(
        volume,
        use_cnn_filter=False,
        config_params=GALLERY_DETECTOR_CONFIG,
    )

    assert result.n_divisions() >= 1


def test_sparse_flavor_produces_no_divisions():
    volume = make_synthetic_volume(
        n_frames=8,
        flavor="sparse",
        seed=3,
    )

    result = run_detection_and_tracking(
        volume,
        use_cnn_filter=False,
        config_params=GALLERY_DETECTOR_CONFIG,
    )

    assert result.n_divisions() == 0


def test_all_flavors_stay_within_canvas_bounds():
    """Regression test for the out-of-bounds drift bug found while building this — cells
    should never be placed outside [0, shape) after clip_to_canvas."""
    for flavor in ["sparse", "crowded", "division_rich", "default"]:
        volume = make_synthetic_volume(
            n_frames=8,
            shape=(96, 96),
            flavor=flavor,
            seed=1,
        )

        result = run_detection_and_tracking(
            volume,
            use_cnn_filter=False,
            config_params=GALLERY_DETECTOR_CONFIG,
        )

        for n in result.nodes:
            assert 0 <= n.y < 96
            assert 0 <= n.x < 96