import random

import numpy as np
import torch

from src.seed import seed_everything


def test_seed_everything_reproducible_across_libraries():
    seed_everything(123, deterministic=False)  # deterministic=True is slow/CPU-only edge cases; skip for speed
    a_random = random.random()
    a_numpy = np.random.rand()
    a_torch = torch.rand(3)

    seed_everything(123, deterministic=False)
    b_random = random.random()
    b_numpy = np.random.rand()
    b_torch = torch.rand(3)

    assert a_random == b_random
    assert a_numpy == b_numpy
    assert torch.equal(a_torch, b_torch)


def test_different_seeds_diverge():
    seed_everything(1, deterministic=False)
    a = np.random.rand()

    seed_everything(2, deterministic=False)
    b = np.random.rand()

    assert a != b
