"""
Training and inference engine.

Implements the optimisation procedure of Table III / Algorithm 1:
  * AdamW, lr = 3e-4, weight decay = 1e-4
  * ReduceLROnPlateau on validation balanced accuracy, factor 0.2, patience 3
  * effective batch size 16 (micro-batch 8 x 2 gradient accumulation)
  * mixed-precision training; FP32 inference so logits cannot overflow
  * early stopping on balanced accuracy, best checkpoint restored

and the test-time augmentation of Sec. IV-A: softmax outputs of the original
image and its +3 deg / -3 deg rotations are averaged.
"""

from __future__ import annotations

import copy
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from .config import EvalConfig, TrainConfig


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model: nn.Module,
            loader: DataLoader,
            device: torch.device,
            use_tta: bool = False,
            tta_angles: Sequence[float] = (0.0, 3.0, -3.0)) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob).  Inference runs in FP32 with NaN guarding."""
    model.eval()
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    angles = tuple(tta_angles) if use_tta else (0.0,)
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        with autocast("cuda", enabled=False):
            logits = None
            for angle in angles:
                rotated = images if angle == 0.0 else TF.rotate(images, float(angle))
                out = model(rotated).float()
                logits = out if logits is None else logits + out
            logits = logits / len(angles)
        p = torch.softmax(torch.nan_to_num(logits), dim=1)
        probs.append(p.cpu().numpy())
        labels.append(target.numpy())

    return np.concatenate(labels), np.concatenate(probs)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                  device: torch.device) -> float:
    """Mean validation loss, used for the loss-vs-epoch curve (Fig. 8b)."""
    model.eval()
    total, batches = 0.0, 0
    for images, target in loader:
        with autocast("cuda", enabled=False):
            logits = model(images.to(device)).float()
        total += float(criterion(logits, target.to(device)).item())
        batches += 1
    return total / max(1, batches)


# ---------------------------------------------------------------------------
# Optimiser / scheduler
# ---------------------------------------------------------------------------
def build_optimizer(model: nn.Module, train_cfg: TrainConfig,
                    head_params: Optional[List[nn.Parameter]] = None,
                    head_lr: Optional[float] = None) -> optim.Optimizer:
    """AdamW as specified in Table III.

    A single learning rate is applied to all parameters, matching the
    manuscript.  Passing `head_lr` enables an optional discriminative rate for
    the attention module and classifier.
    """
    if head_params is None or head_lr is None:
        return optim.AdamW(model.parameters(), lr=train_cfg.lr,
                           weight_decay=train_cfg.weight_decay)
    head_ids = {id(p) for p in head_params}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return optim.AdamW([
        {"params": backbone, "lr": train_cfg.lr},
        {"params": head_params, "lr": head_lr},
    ], weight_decay=train_cfg.weight_decay)


def build_scheduler(optimizer: optim.Optimizer, train_cfg: TrainConfig):
    """ReduceLROnPlateau on validation balanced accuracy (Table III)."""
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                mode="max",
                                                factor=train_cfg.scheduler_factor,
                                                patience=train_cfg.scheduler_patience)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                criterion: nn.Module,
                device: torch.device,
                train_cfg: TrainConfig,
                eval_cfg: EvalConfig,
                head_params: Optional[List[nn.Parameter]] = None,
                backbone_params: Optional[List[nn.Parameter]] = None,
                log_fn: Optional[Callable[[str], None]] = None,
                tag: str = "") -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train one model and return (best model, history).

    The best checkpoint is selected on validation balanced accuracy, the metric
    the manuscript ranks by.
    """
    log = log_fn or (lambda msg: None)
    model = model.to(device)

    optimizer = build_optimizer(model, train_cfg, head_params=None, head_lr=None)
    scheduler = build_scheduler(optimizer, train_cfg)
    scaler = GradScaler("cuda", enabled=(train_cfg.amp and device.type == "cuda"))
    plain_ce = nn.CrossEntropyLoss()

    # Optional warm-up: keep the pretrained backbone frozen for the first epochs
    if backbone_params and train_cfg.freeze_backbone_epochs > 0:
        for p in backbone_params:
            p.requires_grad = False

    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history: Dict[str, List[float]] = {"train_acc": [], "val_acc": [],
                                       "train_loss": [], "val_loss": [],
                                       "val_balanced_acc": [], "lr": []}
    start = time.time()

    for epoch in range(train_cfg.epochs):
        if backbone_params and epoch == train_cfg.freeze_backbone_epochs:
            for p in backbone_params:
                p.requires_grad = True

        model.train()
        correct = total = 0
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (images, target) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with autocast("cuda", enabled=(train_cfg.amp and device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, target) / train_cfg.grad_accumulation
            scaler.scale(loss).backward()

            is_last = (step + 1) == len(train_loader)
            if (step + 1) % train_cfg.grad_accumulation == 0 or is_last:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item()) * train_cfg.grad_accumulation
            correct += int(logits.argmax(1).eq(target).sum().item())
            total += int(target.size(0))

        # Validation folds hold unmodified originals; TTA matches test-time use.
        y_true, y_prob = predict(model, val_loader, device,
                                 use_tta=eval_cfg.use_tta,
                                 tta_angles=eval_cfg.tta_angles)
        y_pred = y_prob.argmax(1)
        balanced = float(balanced_accuracy_score(y_true, y_pred))
        scheduler.step(balanced)

        history["train_acc"].append(100.0 * correct / max(1, total))
        history["val_acc"].append(100.0 * float(accuracy_score(y_true, y_pred)))
        history["val_balanced_acc"].append(100.0 * balanced)
        history["train_loss"].append(running_loss / max(1, len(train_loader)))
        history["val_loss"].append(evaluate_loss(model, val_loader, plain_ce, device))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            log(f"    [{tag}] epoch {epoch+1:02d}/{train_cfg.epochs} "
                f"train {history['train_acc'][-1]:.1f}% | "
                f"val {history['val_acc'][-1]:.1f}% | "
                f"val bal {balanced*100:.2f}%")

        if balanced > best_score:
            best_score = balanced
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg.early_stopping_patience:
                log(f"    [{tag}] early stopping at epoch {epoch+1} "
                    f"(best balanced accuracy {best_score*100:.2f}%)")
                break

    model.load_state_dict(best_state)
    history["best_balanced_acc"] = [best_score * 100.0]
    history["train_seconds"] = [time.time() - start]
    return model, history


# ---------------------------------------------------------------------------
# Complexity profiling — Table VI (and Tables VII, VIII)
# ---------------------------------------------------------------------------
def measure_complexity(model: nn.Module,
                       input_size: int,
                       device: torch.device,
                       eval_cfg: Optional[EvalConfig] = None) -> Dict[str, float]:
    """Parameters (M), GFLOPs and per-image latency (ms).

    Protocol from Sec. IV-B: batch size 8, four warm-up iterations followed by
    20 timed forward passes (reported on a Tesla T4).
    """
    eval_cfg = eval_cfg or EvalConfig()
    model = model.to(device).eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    gflops = float("nan")
    try:
        from thop import profile
        macs, _ = profile(model,
                          inputs=(torch.randn(1, 3, input_size, input_size).to(device),),
                          verbose=False)
        gflops = macs * 2 / 1e9
    except Exception:                      # thop is optional
        pass

    batch = torch.randn(eval_cfg.latency_batch, 3, input_size, input_size).to(device)
    with torch.no_grad():
        for _ in range(eval_cfg.latency_warmup):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(eval_cfg.latency_repeats):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

    ms_per_image = elapsed / eval_cfg.latency_repeats / eval_cfg.latency_batch * 1000.0
    return {"params_m": params_m, "gflops": gflops, "ms_per_image": ms_per_image}


def measure_memory_footprint(model: nn.Module, input_size: int,
                             device: torch.device, batch: int = 1) -> List[Dict]:
    """Forward-pass tensor sizes reproducing Table VIII."""
    from .models import ECANet

    if not isinstance(model, ECANet):
        raise TypeError("Memory footprint table is defined for ECA-Net.")
    model = model.to(device).eval()
    rows = []
    with torch.no_grad():
        x = torch.randn(batch, 3, input_size, input_size, device=device)
        rows.append({"Stage": "Input", "Shape": tuple(x.shape),
                     "Bytes_FP32": x.numel() * 4})
        f = model.features(x)
        rows.append({"Stage": "After Backbone", "Shape": tuple(f.shape),
                     "Bytes_FP32": f.numel() * 4})
        if model.attention.channel_attention is not None:
            ca = model.attention.channel_attention(f)
            rows.append({"Stage": "Channel Attention", "Shape": tuple(ca.shape),
                         "Bytes_FP32": ca.numel() * 4})
        out = model(x)
        rows.append({"Stage": "Final Output", "Shape": tuple(out.shape),
                     "Bytes_FP32": out.numel() * 4})
    return rows
