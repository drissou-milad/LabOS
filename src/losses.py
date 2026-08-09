"""
Alternative loss functions. Selected via config.LOSS_FN ("bce" or "focal")
and instantiated by get_criterion() below.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config


class FocalLoss(nn.Module):
    """
    Binary focal loss, operating on raw logits (mirrors BCEWithLogitsLoss's
    numerically stable log-sum-exp formulation rather than applying a
    separate Sigmoid + log, which would reintroduce the saturation problem
    this project already hit once).

    gamma: focusing parameter (higher = more down-weighting of easy examples)
    alpha: weight for the positive class (use 0.5 for no class re-weighting)
    """

    def __init__(self, gamma=None, alpha=None):
        super().__init__()
        self.gamma = config.FOCAL_GAMMA if gamma is None else gamma
        self.alpha = config.FOCAL_ALPHA if alpha is None else alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)  # = p if target==1 else (1-p), recovered stably
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_term = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal_term.mean()


def get_criterion(name=None):
    """Return the configured loss module. Both options expect raw logits."""
    name = (name or config.LOSS_FN).lower()

    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "focal":
        return FocalLoss()

    raise ValueError(f"Unknown LOSS_FN '{name}', expected 'bce' or 'focal'")
