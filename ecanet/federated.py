"""
ECA-FedNet: federated extension of ECA-Net (Sec. III-G, Algorithms 2 and 3).

Server (Algorithm 2)
    for t = 0 .. T-1:
        every client k receives w(t), runs ClientUpdate and returns (w_k, N_k)
        floating-point tensors are averaged with weight N_k / N
        integer buffers (BN counters) are copied from the first client
        w(t+1) is evaluated on the held-out test set

Client (Algorithm 3)
    local class counts n_c -> weights alpha_c proportional to 1/sqrt(n_c),
    normalised so they sum to C; AdamW; cross-entropy with label smoothing 0.1;
    E local epochs over the private dataset.

Only model parameters cross the client-server boundary — no gradients, no
activations and no raw ECG images.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .config import FederatedConfig
from .losses import InverseSqrtWeightedCE

StateDict = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Server-side aggregation — Algorithm 2, Eq. (8) / (10)
# ---------------------------------------------------------------------------
def fedavg(states: Sequence[StateDict],
           sizes: Sequence[int],
           integer_from_first_client: bool = True) -> StateDict:
    """Sample-weighted FedAvg over floating-point tensors.

    Integer-valued buffers (chiefly `num_batches_tracked`) are not averaged;
    they are taken from the first client, as stated in Sec. III-G.
    """
    if not states:
        raise ValueError("No client states to aggregate.")
    total = float(sum(sizes))
    if total <= 0:
        raise ValueError("Total client sample count must be positive.")

    aggregated = copy.deepcopy(states[0])
    for key in aggregated:
        if aggregated[key].dtype.is_floating_point:
            stacked = sum(states[i][key].to(torch.float64) * (sizes[i] / total)
                          for i in range(len(states)))
            aggregated[key] = stacked.to(states[0][key].dtype)
        elif not integer_from_first_client:
            stacked = sum(states[i][key].to(torch.float64) * (sizes[i] / total)
                          for i in range(len(states)))
            aggregated[key] = stacked.round().to(states[0][key].dtype)
        # else: keep states[0][key]
    return aggregated


def to_cpu_state(model: nn.Module) -> StateDict:
    """Detached CPU copy of the model state — what a client would transmit."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def communication_cost_mb(state: StateDict) -> float:
    """Payload size of one client-to-server transmission, in megabytes."""
    total_bytes = sum(v.numel() * v.element_size() for v in state.values())
    return total_bytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# Client-side update — Algorithm 3
# ---------------------------------------------------------------------------
def client_update(model_factory: Callable[[], nn.Module],
                  global_state: StateDict,
                  loader: DataLoader,
                  class_counts: np.ndarray,
                  device: torch.device,
                  fed_cfg: FederatedConfig,
                  num_samples: int,
                  amp: bool = True) -> Tuple[StateDict, int]:
    """Run E local epochs starting from the global parameters.

    Returns the locally updated state dictionary and the client's sample count,
    which the server uses as the FedAvg weight.
    """
    model = model_factory().to(device)
    model.load_state_dict({k: v.to(device) for k, v in global_state.items()})

    criterion = InverseSqrtWeightedCE(class_counts,
                                      label_smoothing=fed_cfg.label_smoothing,
                                      device=device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=fed_cfg.local_lr,
                            weight_decay=fed_cfg.local_weight_decay)
    scaler = GradScaler("cuda", enabled=(amp and device.type == "cuda"))

    model.train()
    for _ in range(fed_cfg.local_epochs):
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(amp and device.type == "cuda")):
                loss = criterion(model(images), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    state = to_cpu_state(model)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state, int(num_samples)


def select_participants(num_clients: int, participation: float,
                        rng: Optional[np.random.Generator] = None) -> List[int]:
    """Client sampling.  The manuscript uses 100 % participation every round."""
    if participation >= 1.0:
        return list(range(num_clients))
    rng = rng or np.random.default_rng(42)
    k = max(1, int(round(num_clients * participation)))
    return sorted(rng.choice(num_clients, size=k, replace=False).tolist())
