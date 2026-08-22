"""
Loss functions.

Centralized training (Sec. III-E, Algorithm 1 line 13)
    Class-balanced focal loss with effective-number weighting (beta = 0.99),
    focusing parameter gamma = 2.0 and label smoothing eps = 0.05.

Federated local training (Sec. III-G, Algorithm 3 line 10)
    Cross-entropy weighted by the inverse square root of the *local* class
    frequency, with label smoothing eps = 0.1.  Weights are recomputed on each
    client from its own data; no client shares its class distribution.

Table III of the manuscript describes the centralized loss as plain
cross-entropy with eps = 0.1.  Both are implemented; the switch lives in
`config.CENTRAL_LOSS`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedFocalLoss(nn.Module):
    """Class-balanced focal loss, CB(p_t) = alpha_c * (1 - p_t)^gamma * CE.

    alpha_c follows the effective-number formulation
        alpha_c ∝ (1 - beta) / (1 - beta^{n_c}),
    normalised so the weights sum to the number of classes.
    """

    def __init__(self,
                 class_counts: Sequence[float],
                 beta: float = 0.99,
                 gamma: float = 2.0,
                 label_smoothing: float = 0.05,
                 device: Optional[torch.device] = None):
        super().__init__()
        counts = np.asarray(class_counts, dtype=float)
        effective = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.clip(effective, 1e-8, None)
        weights = weights / weights.sum() * len(counts)
        self.register_buffer("weight",
                             torch.tensor(weights, dtype=torch.float, device=device))
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target,
                             weight=self.weight.to(logits.device),
                             label_smoothing=self.label_smoothing,
                             reduction="none")
        p_t = torch.exp(-ce)
        return ((1.0 - p_t) ** self.gamma * ce).mean()


class InverseSqrtWeightedCE(nn.Module):
    """Cross-entropy weighted by 1 / sqrt(n_c), normalised to sum to C.

    This is the local objective used by every federated client (Algorithm 3,
    lines 3-4 and 10) and the alternative centralized loss of Table III.
    """

    def __init__(self,
                 class_counts: Sequence[float],
                 label_smoothing: float = 0.1,
                 device: Optional[torch.device] = None):
        super().__init__()
        counts = np.asarray(class_counts, dtype=float)
        weights = 1.0 / np.sqrt(np.clip(counts, 1.0, None))
        weights = weights / weights.sum() * len(counts)
        self.register_buffer("weight",
                             torch.tensor(weights, dtype=torch.float, device=device))
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target,
                               weight=self.weight.to(logits.device),
                               label_smoothing=self.label_smoothing)


def build_loss(name: str,
               class_counts: Sequence[float],
               device: Optional[torch.device] = None,
               beta: float = 0.99,
               gamma: float = 2.0,
               focal_label_smoothing: float = 0.05,
               ce_label_smoothing: float = 0.1) -> nn.Module:
    """Factory used by the training scripts.

    `class_counts` must be computed from the training fold or the local client
    partition only.
    """
    if name == "cb_focal":
        return ClassBalancedFocalLoss(class_counts, beta, gamma,
                                      focal_label_smoothing, device)
    if name == "weighted_ce":
        return InverseSqrtWeightedCE(class_counts, ce_label_smoothing, device)
    raise ValueError(f"Unknown loss '{name}'. Use 'cb_focal' or 'weighted_ce'.")


def class_counts_from_targets(targets: np.ndarray,
                              indices: Sequence[int],
                              num_classes: int) -> np.ndarray:
    """Count class occurrences within a training subset."""
    return np.bincount(np.asarray(targets)[list(indices)],
                       minlength=num_classes).astype(float)
