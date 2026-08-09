"""
Single source of truth for reproducibility. Call seed_everything() once at
the start of a run, before building datasets/model/optimizer — every other
place that used to call torch.manual_seed()/np.random.seed() individually
should call this instead, so "reproducible" actually means one function,
not three call sites that can drift out of sync.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed=42, deterministic=True):
    """
    Seed Python's `random`, NumPy, and PyTorch (CPU + all CUDA devices).

    deterministic=True also asks cuDNN to use deterministic algorithms.
    This can slow training down slightly and will raise instead of
    silently falling back for a few ops that have no deterministic
    implementation — set deterministic=False if you hit that and don't
    need bitwise-identical reruns.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if CUDA isn't available

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # use_deterministic_algorithms can raise on ops without a
        # deterministic kernel; opt in but don't crash the whole run over it
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    return seed
