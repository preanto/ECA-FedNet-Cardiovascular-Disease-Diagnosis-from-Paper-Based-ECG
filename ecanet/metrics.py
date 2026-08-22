"""
Evaluation metrics for every results table in the manuscript.

Reported quantities (Tables X-XIII, Sec. IV-A "Uncertainty quantification"):
  * raw accuracy, balanced accuracy
  * macro-F1, Cohen's kappa, MCC, G-mean
  * per-class precision, recall, specificity, F1, NPV, support
  * ROC-AUC (binary) or macro one-vs-rest ROC-AUC (three-class)
  * 95 % confidence intervals for accuracy, balanced accuracy and the per-class
    proportions.

Confidence intervals
--------------------
The manuscript states that intervals are "estimated from the observed
proportions and corresponding sample sizes", and that no intervals are given
for macro-F1, kappa, MCC or G-mean because those metrics have no closed-form
binomial variance.  This module implements both closed-form options:

    ci_method='wilson'  Wilson score interval (default; better small-sample
                        coverage, which matters for MI with n = 222)
    ci_method='normal'  Wald / normal-approximation interval

Balanced accuracy is the unweighted mean of the per-class recalls, so its
interval is obtained by propagating the per-class variances
(Var = sum_c p_c (1 - p_c) / n_c / C^2).  A percentile bootstrap is also
provided (`ci_method='bootstrap'`) for anyone who prefers a resampling estimate.
Small differences from the published intervals are expected between these
estimators; the point estimates are unaffected.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, average_precision_score)


# ---------------------------------------------------------------------------
# Interval estimators
# ---------------------------------------------------------------------------
def _z(level: float = 0.95) -> float:
    return float(norm.ppf(0.5 + level / 2.0))


def wilson_ci(successes: float, n: int, level: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z = _z(level)
    p = successes / n
    denom = 1.0 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def normal_ci(successes: float, n: int, level: float = 0.95) -> Tuple[float, float]:
    """Wald interval for a binomial proportion."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z = _z(level)
    p = successes / n
    margin = z * np.sqrt(p * (1 - p) / n)
    return (max(0.0, p - margin), min(1.0, p + margin))


def proportion_ci(successes: float, n: int, method: str = "wilson",
                  level: float = 0.95) -> Tuple[float, float]:
    if method == "wilson":
        return wilson_ci(successes, n, level)
    if method == "normal":
        return normal_ci(successes, n, level)
    raise ValueError(f"Unknown closed-form CI method: {method}")


def balanced_accuracy_ci(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int,
                         level: float = 0.95) -> Tuple[float, float]:
    """Delta-method interval for the unweighted mean of per-class recalls."""
    z = _z(level)
    variance = 0.0
    recalls = []
    for c in range(num_classes):
        n_c = int((y_true == c).sum())
        if n_c == 0:
            continue
        p_c = float(((y_true == c) & (y_pred == c)).sum()) / n_c
        recalls.append(p_c)
        variance += p_c * (1 - p_c) / n_c
    k = max(1, len(recalls))
    variance /= k ** 2
    centre = float(np.mean(recalls)) if recalls else float("nan")
    margin = z * float(np.sqrt(variance))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 metric_fn: Callable[[np.ndarray, np.ndarray], float],
                 n_bootstrap: int = 2000, level: float = 0.95,
                 seed: int = 42) -> Tuple[float, float, float]:
    """Percentile bootstrap: returns (mean, low, high)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        try:
            values.append(metric_fn(y_true[idx], y_pred[idx]))
        except ValueError:            # a resample may miss a class entirely
            continue
    if not values:
        return (float("nan"),) * 3
    alpha = (1.0 - level) / 2.0 * 100
    return (float(np.mean(values)),
            float(np.percentile(values, alpha)),
            float(np.percentile(values, 100 - alpha)))


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def geometric_mean_recall(y_true: np.ndarray, y_pred: np.ndarray,
                          num_classes: int) -> float:
    """G-mean: geometric mean of the per-class recalls (Table XI)."""
    recalls = []
    for c in range(num_classes):
        n_c = int((y_true == c).sum())
        if n_c == 0:
            continue
        recalls.append(float(((y_true == c) & (y_pred == c)).sum()) / n_c)
    if not recalls:
        return float("nan")
    return float(np.exp(np.mean(np.log(np.clip(recalls, 1e-12, None)))))


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      class_names: Sequence[str],
                      ci_method: str = "wilson",
                      level: float = 0.95) -> List[Dict]:
    """Per-class precision / recall / specificity / F1 / NPV with intervals.

    Matches the layout of Tables XII and XIII.
    """
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    rows = []
    for c, name in enumerate(class_names):
        tp = int(cm[c, c])
        fn = int(cm[c].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        rows.append({
            "Class": name,
            "Precision": precision,
            "Precision_CI": proportion_ci(tp, tp + fp, ci_method, level),
            "Recall": recall,
            "Recall_CI": proportion_ci(tp, tp + fn, ci_method, level),
            "Specificity": specificity,
            "Specificity_CI": proportion_ci(tn, tn + fp, ci_method, level),
            "NPV": npv,
            "F1": f1,
            "Support": int(cm[c].sum()),
        })
    return rows


def compute_auc(y_true: np.ndarray, y_prob: np.ndarray, mode: str,
                positive_index: int) -> float:
    """ROC-AUC (binary) or macro one-vs-rest ROC-AUC (three-class)."""
    try:
        if mode == "binary":
            return float(roc_auc_score(y_true == positive_index, y_prob[:, positive_index]))
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def compute_average_precision(y_true: np.ndarray, y_prob: np.ndarray, mode: str,
                              positive_index: int, num_classes: int) -> float:
    try:
        if mode == "binary":
            return float(average_precision_score(y_true == positive_index,
                                                 y_prob[:, positive_index]))
        onehot = np.eye(num_classes)[y_true]
        return float(np.mean([average_precision_score(onehot[:, c], y_prob[:, c])
                              for c in range(num_classes)]))
    except ValueError:
        return float("nan")


def expected_calibration_error(y_prob: np.ndarray, y_true: np.ndarray,
                               n_bins: int = 10) -> float:
    """ECE with equal-width confidence bins (reported as a diagnostic)."""
    confidence = y_prob.max(axis=1)
    prediction = y_prob.argmax(axis=1)
    correct = (prediction == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidence > edges[i]) & (confidence <= edges[i + 1])
        if mask.sum() > 0:
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def summarize(y_true: np.ndarray,
              y_pred: np.ndarray,
              y_prob: Optional[np.ndarray],
              class_names: Sequence[str],
              positive_index: int,
              mode: str = "three_class",
              ci_method: str = "wilson",
              level: float = 0.95,
              n_bootstrap: int = 2000,
              seed: int = 42) -> Dict:
    """One dictionary containing every metric the manuscript reports."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num_classes = len(class_names)
    n = len(y_true)

    accuracy = float(accuracy_score(y_true, y_pred))
    balanced = float(balanced_accuracy_score(y_true, y_pred))

    if ci_method == "bootstrap":
        _, acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, accuracy_score,
                                         n_bootstrap, level, seed)
        _, bal_lo, bal_hi = bootstrap_ci(y_true, y_pred, balanced_accuracy_score,
                                         n_bootstrap, level, seed)
        per_class_method = "wilson"
    else:
        acc_lo, acc_hi = proportion_ci(accuracy * n, n, ci_method, level)
        bal_lo, bal_hi = balanced_accuracy_ci(y_true, y_pred, num_classes, level)
        per_class_method = ci_method

    result = {
        "n": int(n),
        "accuracy": accuracy,
        "accuracy_ci": (acc_lo, acc_hi),
        "balanced_accuracy": balanced,
        "balanced_accuracy_ci": (bal_lo, bal_hi),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro",
                                                 zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro",
                                           zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "g_mean": geometric_mean_recall(y_true, y_pred, num_classes),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(num_classes))).tolist(),
        "per_class": per_class_metrics(y_true, y_pred, class_names,
                                       per_class_method, level),
        "class_names": list(class_names),
    }

    # Sensitivity of the clinically critical class (MI, or Abnormal in binary mode)
    pos_support = int((y_true == positive_index).sum())
    pos_hits = int(((y_true == positive_index) & (y_pred == positive_index)).sum())
    result["positive_class"] = class_names[positive_index]
    result["positive_sensitivity"] = pos_hits / pos_support if pos_support else float("nan")
    result["positive_sensitivity_ci"] = proportion_ci(pos_hits, pos_support,
                                                      per_class_method, level)
    # "MI missed entirely" column of Table XI: true MI predicted as Normal
    normal_index = class_names.index("Normal") if "Normal" in class_names else None
    if normal_index is not None and pos_support:
        missed = int(((y_true == positive_index) & (y_pred == normal_index)).sum())
        result["positive_missed_as_normal"] = missed
        result["positive_missed_fraction"] = missed / pos_support

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        result["auc"] = compute_auc(y_true, y_prob, mode, positive_index)
        result["average_precision"] = compute_average_precision(
            y_true, y_prob, mode, positive_index, num_classes)
        onehot = np.eye(num_classes)[y_true]
        result["brier"] = float(np.mean(np.sum((y_prob - onehot) ** 2, axis=1)))
        result["ece"] = expected_calibration_error(y_prob, y_true)

    return result


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def format_summary(summary: Dict, title: str = "Results") -> str:
    """Render a summary dictionary as the tables appear in the manuscript."""
    lines = ["", "=" * 78, title, "=" * 78]
    acc_lo, acc_hi = summary["accuracy_ci"]
    bal_lo, bal_hi = summary["balanced_accuracy_ci"]
    lines.append(f"  n                  : {summary['n']}")
    lines.append(f"  Raw accuracy       : {summary['accuracy']*100:.2f}% "
                 f"[{acc_lo*100:.2f}-{acc_hi*100:.2f}]")
    lines.append(f"  Balanced accuracy  : {summary['balanced_accuracy']*100:.2f}% "
                 f"[{bal_lo*100:.2f}-{bal_hi*100:.2f}]")
    lines.append(f"  Macro-F1           : {summary['macro_f1']:.4f}")
    lines.append(f"  Cohen's kappa      : {summary['cohen_kappa']:.4f}")
    lines.append(f"  MCC                : {summary['mcc']:.4f}")
    lines.append(f"  G-mean             : {summary['g_mean']:.4f}")
    if "auc" in summary:
        lines.append(f"  ROC-AUC            : {summary['auc']:.4f}")
    pos = summary.get("positive_class", "")
    if "positive_sensitivity" in summary:
        lo, hi = summary["positive_sensitivity_ci"]
        lines.append(f"  {pos} sensitivity  : {summary['positive_sensitivity']*100:.2f}% "
                     f"[{lo*100:.2f}-{hi*100:.2f}]")
    if "positive_missed_as_normal" in summary:
        lines.append(f"  {pos} predicted Normal: {summary['positive_missed_as_normal']} "
                     f"({summary['positive_missed_fraction']*100:.1f}%)")

    lines.append("")
    name_width = max(12, max(len(row["Class"]) for row in summary["per_class"]) + 2)
    lines.append(f"  {'Class':<{name_width}}{'Prec.(%)':>22}{'Recall(%)':>22}"
                 f"{'Spec.(%)':>22}{'F1(%)':>10}{'Support':>9}")

    def _cell(value: float, interval) -> str:
        lo, hi = interval
        if not np.isfinite(lo) or not np.isfinite(hi):
            return f"{value*100:>8.2f} {'[undefined]':>13}"
        return f"{value*100:>8.2f} [{lo*100:5.2f}-{hi*100:5.2f}]"

    for row in summary["per_class"]:
        lines.append(
            f"  {row['Class']:<{name_width}}"
            + _cell(row["Precision"], row["Precision_CI"])
            + _cell(row["Recall"], row["Recall_CI"])
            + _cell(row["Specificity"], row["Specificity_CI"])
            + f"{row['F1']*100:>10.2f}{row['Support']:>9d}")

    lines.append("")
    lines.append("  Confusion matrix (rows = true, columns = predicted):")
    label_width = max(12, max(len(n) for n in summary["class_names"]) + 2)
    header = "".join(f"{name[:9]:>10}" for name in summary["class_names"])
    lines.append(f"  {'':<{label_width}}{header}")
    for name, row in zip(summary["class_names"], summary["confusion_matrix"]):
        lines.append(f"  {name:<{label_width}}" + "".join(f"{v:>10d}" for v in row))
    lines.append("=" * 78)
    return "\n".join(lines)
