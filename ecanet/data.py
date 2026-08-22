"""
Dataset construction, augmentation (Table II), class balancing (Sec. III-E) and
the two evaluation protocols of Sec. IV-A.

Protocol A — stratified 5-fold cross-validation over the 1,000 original records;
             augmentation is applied to the training portion of each fold only,
             validation folds contain unmodified originals, and the five sets of
             out-of-fold predictions are pooled.
Protocol B — a single stratified 80/20 split (seed 42).  The 200 test records
             are removed before the federated clients are formed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import AugmentationConfig, DataConfig, EvalConfig, FederatedConfig
from .preprocessing import crop_to_ink


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------
@dataclass
class DatasetIndex:
    """Lightweight description of the dataset shared by every experiment."""

    samples: List[Tuple[str, int]]     # (path, raw_label)
    targets: np.ndarray               # task labels after binary/three-class mapping
    class_names: List[str]
    positive_index: int               # MI (three-class) or Abnormal (binary)
    raw_classes: List[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def size(self) -> int:
        return len(self.samples)

    def counts(self) -> dict:
        return {name: int((self.targets == i).sum()) for i, name in enumerate(self.class_names)}


def build_index(cfg: DataConfig) -> DatasetIndex:
    """Scan an ImageFolder-style directory and apply the task label mapping.

    Folder names are sorted alphabetically by torchvision, giving
    Abnormal = 0, MI = 1, Normal = 2 — the class order used in the manuscript
    tables.
    """
    base = datasets.ImageFolder(cfg.root)
    samples = list(base.samples)
    raw_classes = list(base.classes)
    raw_targets = np.array([label for _, label in samples])

    if cfg.normal_dir_name not in raw_classes:
        raise ValueError(
            f"Expected a '{cfg.normal_dir_name}' class directory under {cfg.root}; "
            f"found {raw_classes}."
        )
    normal_idx = raw_classes.index(cfg.normal_dir_name)

    if cfg.mode == "binary":
        targets = (raw_targets != normal_idx).astype(int)
        class_names = ["Normal", "Abnormal"]
        positive = 1
    elif cfg.mode == "three_class":
        targets = raw_targets.copy()
        class_names = list(raw_classes)
        positive = raw_classes.index("MI") if "MI" in raw_classes else 0
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    return DatasetIndex(samples=samples, targets=targets, class_names=class_names,
                        positive_index=positive, raw_classes=raw_classes)


# ---------------------------------------------------------------------------
# Transforms — Table II
# ---------------------------------------------------------------------------
def build_transforms(input_size: int, aug: AugmentationConfig) -> Tuple[A.Compose, A.Compose]:
    """Return (train_transform, eval_transform).

    Only the training transform performs augmentation; validation and test
    samples pass through resize + normalise, so no augmented copy of an
    evaluation record is ever created (Sec. III-E).
    """
    normalize = A.Normalize(mean=aug.mean, std=aug.std)
    train_tf = A.Compose([
        A.Resize(input_size, input_size),
        A.ShiftScaleRotate(shift_limit=aug.shift_limit,
                           scale_limit=aug.scale_limit,
                           rotate_limit=aug.rotate_limit,
                           p=aug.ssr_prob,
                           border_mode=cv2.BORDER_REFLECT),
        A.RandomBrightnessContrast(brightness_limit=aug.brightness_limit,
                                   contrast_limit=aug.contrast_limit,
                                   p=aug.brightness_contrast_prob),
        A.CLAHE(p=aug.clahe_prob),
        A.GaussianBlur(blur_limit=aug.gauss_blur_limit, p=aug.gauss_blur_prob),
        A.GaussNoise(var_limit=aug.gauss_noise_var_limit, p=aug.gauss_noise_prob),
        normalize,
        ToTensorV2(),
    ])
    eval_tf = A.Compose([
        A.Resize(input_size, input_size),
        normalize,
        ToTensorV2(),
    ])
    return train_tf, eval_tf


class ECGDataset(Dataset):
    """Reads images by global index so every split shares one file list."""

    def __init__(self,
                 index: DatasetIndex,
                 indices: Sequence[int],
                 transform: A.Compose,
                 crop_ink: bool = True,
                 gray_threshold: int = 245,
                 pad_frac: float = 0.03):
        self.index = index
        self.indices = list(indices)
        self.transform = transform
        self.crop_ink = crop_ink
        self.gray_threshold = gray_threshold
        self.pad_frac = pad_frac

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        gi = self.indices[i]
        path = self.index.samples[gi][0]
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.crop_ink:
            img = crop_to_ink(img, self.gray_threshold, self.pad_frac)
        img = self.transform(image=img)["image"]
        return img, int(self.index.targets[gi])


# ---------------------------------------------------------------------------
# Class balancing — Sec. III-E
# ---------------------------------------------------------------------------
def inverse_sqrt_class_weights(targets: np.ndarray,
                               indices: Sequence[int],
                               num_classes: int) -> np.ndarray:
    """Weights proportional to 1 / sqrt(n_c), normalised so they sum to C.

    Computed separately for every training fold / client, so no information
    from validation data or from another client influences the optimisation.
    """
    counts = np.bincount(np.asarray(targets)[list(indices)], minlength=num_classes).astype(float)
    w = 1.0 / np.sqrt(np.clip(counts, 1.0, None))
    return w / w.sum() * num_classes


def effective_number_weights(counts: np.ndarray, beta: float = 0.99) -> np.ndarray:
    """Class-balanced weights with effective-number re-weighting (Sec. III-E)."""
    counts = np.asarray(counts, dtype=float)
    effective = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / np.clip(effective, 1e-8, None)
    return w / w.sum() * len(counts)


def build_weighted_sampler(targets: np.ndarray,
                           indices: Sequence[int],
                           num_classes: int,
                           seed: int = 42) -> WeightedRandomSampler:
    """Weighted random sampler with per-sample weight 1 / sqrt(class frequency)."""
    labels = np.asarray(targets)[list(indices)]
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    per_sample = (1.0 / np.sqrt(np.clip(counts, 1.0, None)))[labels]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(per_sample, dtype=torch.double),
                                 num_samples=len(per_sample),
                                 replacement=True,
                                 generator=generator)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def make_loader(index: DatasetIndex,
                indices: Sequence[int],
                transform: A.Compose,
                batch_size: int,
                train: bool,
                data_cfg: DataConfig,
                sampler=None) -> DataLoader:
    dataset = ECGDataset(index, indices, transform,
                         crop_ink=data_cfg.crop_to_ink,
                         gray_threshold=data_cfg.ink_crop_gray_threshold,
                         pad_frac=data_cfg.ink_crop_pad_frac)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=(train and sampler is None),
                      sampler=sampler,
                      num_workers=data_cfg.num_workers,
                      pin_memory=torch.cuda.is_available(),
                      drop_last=False)


# ---------------------------------------------------------------------------
# Evaluation protocols — Sec. IV-A
# ---------------------------------------------------------------------------
def protocol_a_folds(index: DatasetIndex, eval_cfg: EvalConfig,
                     seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold over the original records (K = 5, shuffle, seed 42)."""
    skf = StratifiedKFold(n_splits=eval_cfg.k_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(index.size), index.targets))


def protocol_b_split(index: DatasetIndex, eval_cfg: EvalConfig,
                     seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Fixed stratified 80/20 split; the 200 test records stay untouched."""
    train_idx, test_idx = train_test_split(np.arange(index.size),
                                           test_size=eval_cfg.test_fraction,
                                           stratify=index.targets,
                                           random_state=seed)
    return np.asarray(train_idx), np.asarray(test_idx)


# ---------------------------------------------------------------------------
# Federated partitioning — Sec. III-G, Tables V and IX
# ---------------------------------------------------------------------------
def dirichlet_partition(indices: np.ndarray,
                        targets: np.ndarray,
                        num_clients: int,
                        num_classes: int,
                        alpha: float = 0.4,
                        rng: np.random.Generator | None = None) -> List[np.ndarray]:
    """Split `indices` across clients with a per-class Dir(alpha) draw.

    Each class is distributed independently, which produces both label skew and
    quantity skew (Table V reports 48-391 records per client, an 8.1-fold range).
    """
    rng = rng or np.random.default_rng(42)
    clients: List[List[int]] = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        class_idx = np.asarray(indices)[targets[np.asarray(indices)] == c]
        if class_idx.size == 0:
            continue
        class_idx = class_idx.copy()
        rng.shuffle(class_idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        for k, part in enumerate(np.split(class_idx, cuts)):
            clients[k].extend(part.tolist())
    return [np.array(sorted(c), dtype=int) for c in clients]


def site_partition(indices: np.ndarray,
                   samples: Sequence[Tuple[str, int]],
                   pattern: str) -> List[np.ndarray]:
    """Partition by a site identifier parsed from the file path.

    Provided for genuine multi-centre data; the manuscript's experiments use the
    Dirichlet partition because all records originate from a single hospital.
    """
    import re

    sites = []
    for i in indices:
        match = re.search(pattern, samples[i][0])
        sites.append(match.group(1) if match else "unknown")
    sites_arr = np.array(sites)
    return [np.asarray(indices)[sites_arr == u] for u in sorted(set(sites))]


def partition_clients(index: DatasetIndex,
                      train_indices: np.ndarray,
                      fed_cfg: FederatedConfig,
                      seed: int = 42) -> List[np.ndarray]:
    if fed_cfg.partition == "site":
        return site_partition(train_indices, index.samples, fed_cfg.site_regex)
    rng = np.random.default_rng(seed)
    return dirichlet_partition(train_indices, index.targets, fed_cfg.num_clients,
                               index.num_classes, fed_cfg.dirichlet_alpha, rng)


def describe_clients(index: DatasetIndex, clients: Sequence[np.ndarray]) -> List[dict]:
    """Reproduce Table V: client-wise distribution of ECG classes."""
    rows = []
    for k, ci in enumerate(clients, start=1):
        row = {"Client": k, "N_k": int(len(ci))}
        for c, name in enumerate(index.class_names):
            row[name] = int((index.targets[ci] == c).sum()) if len(ci) else 0
        rows.append(row)
    total = {"Client": "Total", "N_k": int(sum(len(c) for c in clients))}
    for c, name in enumerate(index.class_names):
        total[name] = int(sum(int((index.targets[ci] == c).sum()) for ci in clients if len(ci)))
    rows.append(total)
    return rows
