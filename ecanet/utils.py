"""Shared utilities: seeding, device selection, logging and result serialisation."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:  # torch is optional for the pure-numpy utilities (metrics, partitioning)
    import torch
except ImportError:  # pragma: no cover
    torch = None


LOGGER_NAME = "ecanet"


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch, and make cuDNN deterministic.

    Sec. IV-A of the manuscript fixes seed = 42 for both evaluation protocols.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True):
    if torch is None:
        raise ImportError("PyTorch is required for this operation.")
    return torch.device("cuda" if (prefer_cuda and torch.cuda.is_available()) else "cpu")


def setup_logger(output_dir: str | os.PathLike | None = None,
                 name: str = LOGGER_NAME,
                 filename: str = "run.log") -> logging.Logger:
    """Console + optional file logger.  Safe to call repeatedly."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(out / filename)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(obj: Dict[str, Any], path: str | os.PathLike) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, cls=_NumpyEncoder)


def load_json(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def count_parameters(module) -> int:
    """Trainable parameter count (used for Tables IV and VII)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def human_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h {m:02d}m {s:02d}s" if h else f"{m:d}m {s:02d}s"
