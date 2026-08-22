#!/usr/bin/env python
"""
Attention ablation study (Table XII).

Four configurations are trained under the identical Protocol A pipeline:

    M_base   vanilla EfficientNet-B3, no attention
    M_CA     channel attention only
    M_SA     spatial attention only
    M_dual   proposed sequential channel -> spatial module

Because the branches are sequential, the two single-branch variants are not
independent contributions; the manuscript reports that their combined gain is
sub-additive (10.89 points versus the 16.58 expected under independence).

    python scripts/run_ablation.py --data-root data/SSMCH-ECG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecanet.config import ABLATION_VARIANTS, add_common_args, config_from_args
from ecanet.data import (build_index, build_transforms, build_weighted_sampler,
                         make_loader, protocol_a_folds)
from ecanet.engine import predict, train_model
from ecanet.losses import build_loss, class_counts_from_targets
from ecanet.metrics import format_summary, summarize
from ecanet.models import build_ablation_model
from ecanet.utils import ensure_dir, save_json, seed_everything, setup_logger
from ecanet.visualization import plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attention ablation (Table XII).")
    add_common_args(parser)
    parser.add_argument("--variants", nargs="*", default=None,
                        choices=["M_base", "M_CA", "M_SA", "M_dual"])
    return parser.parse_args()


def run_variant(name: str, attention: str, cfg, index, device, logger) -> dict:
    """Train one configuration with pooled out-of-fold evaluation."""
    train_tf, eval_tf = build_transforms(cfg.model.input_size, cfg.augment)
    folds = protocol_a_folds(index, cfg.eval, cfg.seed)

    oof_true, oof_prob = [], []
    for fold, (train_idx, val_idx) in enumerate(folds, start=1):
        seed_everything(cfg.seed + fold)
        sampler = (build_weighted_sampler(index.targets, train_idx, index.num_classes,
                                          cfg.seed)
                   if cfg.train.use_weighted_sampler else None)
        train_loader = make_loader(index, train_idx, train_tf, cfg.train.batch_size,
                                   True, cfg.data, sampler=sampler)
        val_loader = make_loader(index, val_idx, eval_tf, cfg.train.batch_size,
                                 False, cfg.data)

        model = build_ablation_model(attention, index.num_classes, cfg.model)
        backbone_params, head_params = model.parameter_groups()
        counts = class_counts_from_targets(index.targets, train_idx, index.num_classes)
        criterion = build_loss(cfg.train.loss, counts, device,
                               beta=cfg.train.cb_beta,
                               gamma=cfg.train.focal_gamma,
                               focal_label_smoothing=cfg.train.focal_label_smoothing,
                               ce_label_smoothing=cfg.train.ce_label_smoothing)

        model, _ = train_model(model, train_loader, val_loader, criterion, device,
                               cfg.train, cfg.eval,
                               head_params=head_params,
                               backbone_params=backbone_params,
                               log_fn=logger.info, tag=f"{name} fold {fold}")

        y_true, y_prob = predict(model, val_loader, device,
                                 use_tta=cfg.eval.use_tta,
                                 tta_angles=cfg.eval.tta_angles)
        oof_true.append(y_true)
        oof_prob.append(y_prob)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    y_true = np.concatenate(oof_true)
    y_prob = np.concatenate(oof_prob)
    summary = summarize(y_true, y_prob.argmax(1), y_prob, index.class_names,
                        index.positive_index, cfg.data.mode,
                        cfg.eval.ci_method, cfg.eval.ci_level,
                        cfg.eval.n_bootstrap, cfg.seed)
    summary["variant"] = name
    summary["attention"] = attention
    summary["_y_true"] = y_true
    summary["_y_pred"] = y_prob.argmax(1)

    # Attention parameter overhead (Table VII)
    probe = build_ablation_model(attention, index.num_classes, cfg.model)
    summary["parameter_breakdown"] = probe.parameter_breakdown()
    del probe
    return summary


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)

    out_dir = ensure_dir(Path(cfg.output_dir) / f"ablation_{cfg.data.mode}")
    logger = setup_logger(out_dir)
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = build_index(cfg.data)
    logger.info(f"Device: {device} | {index.size} records | counts {index.counts()}")

    variants = ABLATION_VARIANTS
    if args.variants:
        wanted = set(args.variants)
        variants = [v for v in variants if v[0] in wanted]

    rows, summaries = [], {}
    for name, attention in variants:
        logger.info("=" * 70)
        logger.info(f"ABLATION VARIANT: {name} (attention = {attention})")
        seed_everything(cfg.seed)
        summary = run_variant(name, attention, cfg, index, device, logger)

        y_true = summary.pop("_y_true")
        y_pred = summary.pop("_y_pred")
        summaries[name] = summary
        logger.info(format_summary(summary, f"{name} — pooled out-of-fold"))
        plot_confusion_matrix(y_true, y_pred, index.class_names,
                              f"{name} ({cfg.data.mode})", out_dir / f"cm_{name}.png")

        attention_params = (summary["parameter_breakdown"]["Channel Attention"]
                            + summary["parameter_breakdown"]["Spatial Attention"])

        def _cell(value, interval) -> str:
            lo, hi = interval
            if not np.isfinite(lo) or not np.isfinite(hi):
                return f"{value*100:.2f} [undefined]"
            return f"{value*100:.2f} [{lo*100:.2f}-{hi*100:.2f}]"

        for cls in summary["per_class"]:
            rows.append({
                "Model": name,
                "Class": cls["Class"],
                "Precision (%)": _cell(cls["Precision"], cls["Precision_CI"]),
                "Recall (%)": _cell(cls["Recall"], cls["Recall_CI"]),
                "Specificity (%)": _cell(cls["Specificity"], cls["Specificity_CI"]),
                "F1 (%)": f"{cls['F1']*100:.2f}",
                "Support": cls["Support"],
                "Bal. Acc (%)": round(summary["balanced_accuracy"] * 100, 2),
                "Macro-F1 (%)": round(summary["macro_f1"] * 100, 2),
                "Attention params": attention_params,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ablation_table.csv", index=False)
    logger.info("\n" + "=" * 78)
    logger.info(f"ABLATION TABLE ({cfg.data.mode}) — Table XII layout")
    logger.info("=" * 78 + "\n" + df.to_string(index=False))

    # Sub-additivity check discussed in Sec. IV-G
    if {"M_base", "M_CA", "M_SA", "M_dual"} <= set(summaries):
        base = summaries["M_base"]["balanced_accuracy"] * 100
        gain_ca = summaries["M_CA"]["balanced_accuracy"] * 100 - base
        gain_sa = summaries["M_SA"]["balanced_accuracy"] * 100 - base
        gain_dual = summaries["M_dual"]["balanced_accuracy"] * 100 - base
        logger.info("\nAttention branch interaction (Sec. IV-G):")
        logger.info(f"  channel-only gain       : {gain_ca:+.2f} points")
        logger.info(f"  spatial-only gain       : {gain_sa:+.2f} points")
        logger.info(f"  sequential (dual) gain  : {gain_dual:+.2f} points")
        logger.info(f"  additive expectation    : {gain_ca + gain_sa:+.2f} points "
                    f"-> shortfall {gain_ca + gain_sa - gain_dual:.2f} points")

    save_json({"config": cfg.to_dict(), "summaries": summaries},
              out_dir / "results.json")
    logger.info(f"Artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
