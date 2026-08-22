"""
Figure generation: confusion matrices (Fig. 7), accuracy/loss/ROC curves
(Fig. 8), federated convergence (Fig. 9) and Grad-CAM panels (Fig. 10).

All figures are written at 300 dpi to the run's output directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")                      # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             roc_curve, average_precision_score)
from sklearn.preprocessing import label_binarize

DPI = 300


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: Sequence[str], title: str,
                          path: str | Path, normalize: bool = False) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fmt = "d"
    data = cm
    if normalize:
        data = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        fmt = ".2f"
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return _save(fig, path)


def plot_training_curves(histories: Sequence[Dict[str, List[float]]],
                         title: str, path_prefix: str | Path) -> List[Path]:
    """Mean accuracy and loss curves across folds, with a +/-1 std band (Fig. 8a-b)."""
    length = min(len(h["val_acc"]) for h in histories)
    epochs = np.arange(1, length + 1)
    paths = []

    train_acc = np.mean([h["train_acc"][:length] for h in histories], axis=0)
    val_acc = np.mean([h["val_acc"][:length] for h in histories], axis=0)
    val_std = np.std([h["val_acc"][:length] for h in histories], axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(epochs, train_acc, label="Training", lw=2)
    ax.plot(epochs, val_acc, label="Validation", lw=2)
    ax.fill_between(epochs, val_acc - val_std, val_acc + val_std, alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{title} — Accuracy vs Epoch")
    ax.grid(alpha=0.3)
    ax.legend()
    paths.append(_save(fig, f"{path_prefix}_accuracy.png"))

    train_loss = np.mean([h["train_loss"][:length] for h in histories], axis=0)
    val_loss = np.mean([h["val_loss"][:length] for h in histories], axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(epochs, train_loss, label="Training", lw=2)
    ax.plot(epochs, val_loss, label="Validation", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title} — Loss vs Epoch")
    ax.grid(alpha=0.3)
    ax.legend()
    paths.append(_save(fig, f"{path_prefix}_loss.png"))
    return paths


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray,
                    class_names: Sequence[str], title: str,
                    path: str | Path, mode: str = "three_class",
                    positive_index: int = 1) -> Path:
    """Class-wise ROC curves with per-class and macro AUC (Fig. 8c)."""
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    num_classes = len(class_names)

    if mode == "binary":
        fpr, tpr, _ = roc_curve(y_true == positive_index, y_prob[:, positive_index])
        ax.plot(fpr, tpr, lw=2, label=f"{class_names[positive_index]} (AUC = {auc(fpr, tpr):.3f})")
    else:
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        grid = np.linspace(0, 1, 200)
        interpolated = []
        for c, name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
            ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
            interpolated.append(np.interp(grid, fpr, tpr))
        macro = np.mean(interpolated, axis=0)
        ax.plot(grid, macro, "b--", lw=2.2, label=f"Macro (AUC = {auc(grid, macro):.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, path)


def plot_precision_recall(y_true: np.ndarray, y_prob: np.ndarray,
                          class_names: Sequence[str], title: str,
                          path: str | Path, mode: str = "three_class",
                          positive_index: int = 1) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    if mode == "binary":
        precision, recall, _ = precision_recall_curve(y_true == positive_index,
                                                      y_prob[:, positive_index])
        ap = average_precision_score(y_true == positive_index, y_prob[:, positive_index])
        ax.plot(recall, precision, lw=2, label=f"{class_names[positive_index]} (AP = {ap:.3f})")
    else:
        y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        for c, name in enumerate(class_names):
            precision, recall, _ = precision_recall_curve(y_bin[:, c], y_prob[:, c])
            ap = average_precision_score(y_bin[:, c], y_prob[:, c])
            ax.plot(recall, precision, lw=1.8, label=f"{name} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    return _save(fig, path)


def plot_federated_convergence(balanced_per_round: Sequence[float],
                               path: str | Path,
                               centralized_reference: Optional[float] = None,
                               title: str = "ECA-FedNet training convergence") -> Path:
    """Balanced accuracy per communication round (Fig. 9)."""
    rounds = np.arange(1, len(balanced_per_round) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(rounds, np.asarray(balanced_per_round) * 100, marker="o", lw=2,
            label="ECA-FedNet (global model)")
    if centralized_reference is not None:
        ax.axhline(centralized_reference * 100, color="r", ls="--", lw=1.8,
                   label=f"Centralized ({centralized_reference*100:.2f}%)")
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test balanced accuracy (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, path)


def plot_client_distribution(rows: Sequence[Dict], class_names: Sequence[str],
                             path: str | Path,
                             title: str = "Client-wise class distribution") -> Path:
    """Stacked bar chart of the Dirichlet partition (Table V)."""
    clients = [r for r in rows if r["Client"] != "Total"]
    labels = [f"Client {r['Client']}" for r in clients]
    bottom = np.zeros(len(clients))
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for name in class_names:
        values = np.array([r.get(name, 0) for r in clients], dtype=float)
        ax.bar(labels, values, bottom=bottom, label=name)
        bottom += values
    ax.set_ylabel("Number of records")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, path)


def plot_gradcam_panel(images: Sequence[np.ndarray],
                       overlays: Sequence[np.ndarray],
                       captions: Sequence[str],
                       path: str | Path,
                       title: str = "Grad-CAM analysis") -> Path:
    """Original / heat-map pairs for representative records (Fig. 10)."""
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.6), squeeze=False)
    for i in range(n):
        axes[0][i].imshow(images[i])
        axes[0][i].set_title(captions[i], fontsize=10)
        axes[0][i].axis("off")
        axes[1][i].imshow(overlays[i])
        axes[1][i].axis("off")
    axes[0][0].set_ylabel("Input")
    axes[1][0].set_ylabel("Grad-CAM")
    fig.suptitle(title)
    return _save(fig, path)
