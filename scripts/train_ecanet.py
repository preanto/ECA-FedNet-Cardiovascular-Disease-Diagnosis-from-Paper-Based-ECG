#!/usr/bin/env python
"""
Train ECA-Net under Protocol A (Sec. IV-A): stratified five-fold
cross-validation over the 1,000 original records, augmentation restricted to
the training portion of each fold, out-of-fold predictions pooled into a single
set of 1,000 predictions.

Produces the ECA-Net column of Table XI and Figs. 7(g) and 8.

    python scripts/train_ecanet.py --data-root data/SSMCH-ECG --mode three_class
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecanet.config import add_common_args, config_from_args
from ecanet.data import (build_index, build_transforms, build_weighted_sampler,
                         make_loader, protocol_a_folds)
from ecanet.engine import predict, train_model
from ecanet.losses import build_loss, class_counts_from_targets
from ecanet.metrics import format_summary, summarize
from ecanet.models import ECANet
from ecanet.utils import ensure_dir, human_time, save_json, seed_everything, setup_logger
from ecanet.visualization import (plot_confusion_matrix, plot_precision_recall,
                                  plot_roc_curves, plot_training_curves)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ECA-Net — Protocol A (5-fold OOF).")
    add_common_args(parser)
    parser.add_argument("--attention", default="dual",
                        choices=["none", "channel", "spatial", "dual"],
                        help="Attention configuration (Table XII).")
    parser.add_argument("--loss", default=None, choices=["cb_focal", "weighted_ce"],
                        help="Override the centralized loss; default follows config.CENTRAL_LOSS.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    cfg.model.attention = args.attention
    cfg.eval.k_folds = args.folds
    if args.loss:
        cfg.train.loss = args.loss

    out_dir = ensure_dir(Path(cfg.output_dir) / f"ecanet_{cfg.data.mode}_{args.attention}")
    logger = setup_logger(out_dir)
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    index = build_index(cfg.data)
    logger.info(f"Loaded {index.size} records | classes {index.class_names} | "
                f"counts {index.counts()}")

    train_tf, eval_tf = build_transforms(cfg.model.input_size, cfg.augment)
    folds = protocol_a_folds(index, cfg.eval, cfg.seed)

    oof_true, oof_prob = [], []
    fold_order, histories = [], []
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(folds, start=1):
        seed_everything(cfg.seed + fold)
        logger.info(f"===== Fold {fold}/{cfg.eval.k_folds} "
                    f"(train {len(train_idx)}, validation {len(val_idx)}) =====")

        sampler = (build_weighted_sampler(index.targets, train_idx, index.num_classes,
                                          cfg.seed)
                   if cfg.train.use_weighted_sampler else None)
        train_loader = make_loader(index, train_idx, train_tf, cfg.train.batch_size,
                                   True, cfg.data, sampler=sampler)
        val_loader = make_loader(index, val_idx, eval_tf, cfg.train.batch_size,
                                 False, cfg.data)

        model = ECANet(index.num_classes,
                       attention=cfg.model.attention,
                       reduction=cfg.model.reduction_ratio,
                       spatial_kernel=cfg.model.spatial_kernel,
                       dropout=cfg.model.dropout,
                       pretrained=cfg.model.pretrained)
        backbone_params, head_params = model.parameter_groups()

        counts = class_counts_from_targets(index.targets, train_idx, index.num_classes)
        criterion = build_loss(cfg.train.loss, counts, device,
                               beta=cfg.train.cb_beta,
                               gamma=cfg.train.focal_gamma,
                               focal_label_smoothing=cfg.train.focal_label_smoothing,
                               ce_label_smoothing=cfg.train.ce_label_smoothing)

        model, history = train_model(model, train_loader, val_loader, criterion,
                                     device, cfg.train, cfg.eval,
                                     head_params=head_params,
                                     backbone_params=backbone_params,
                                     log_fn=logger.info, tag=f"fold {fold}")
        histories.append(history)

        y_true, y_prob = predict(model, val_loader, device,
                                 use_tta=cfg.eval.use_tta,
                                 tta_angles=cfg.eval.tta_angles)
        oof_true.append(y_true)
        oof_prob.append(y_prob)
        fold_order.append(val_idx)

        fold_summary = summarize(y_true, y_prob.argmax(1), y_prob, index.class_names,
                                 index.positive_index, cfg.data.mode,
                                 cfg.eval.ci_method, cfg.eval.ci_level)
        fold_scores.append(fold_summary["balanced_accuracy"])
        logger.info(f"  fold {fold}: raw {fold_summary['accuracy']*100:.2f}% | "
                    f"balanced {fold_summary['balanced_accuracy']*100:.2f}% | "
                    f"{index.class_names[index.positive_index]} sensitivity "
                    f"{fold_summary['positive_sensitivity']*100:.2f}%")

        if args.save_checkpoints:
            torch.save(model.state_dict(), out_dir / f"ecanet_fold{fold}.pth")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- pooled out-of-fold evaluation (the number reported in Table XI) ----
    y_true = np.concatenate(oof_true)
    y_prob = np.concatenate(oof_prob)
    y_pred = y_prob.argmax(1)

    summary = summarize(y_true, y_pred, y_prob, index.class_names,
                        index.positive_index, cfg.data.mode,
                        cfg.eval.ci_method, cfg.eval.ci_level,
                        cfg.eval.n_bootstrap, cfg.seed)
    summary["per_fold_balanced_accuracy"] = fold_scores
    summary["protocol"] = "A (pooled out-of-fold, 5-fold stratified CV)"
    summary["attention"] = cfg.model.attention
    summary["loss"] = cfg.train.loss
    summary["tta"] = cfg.eval.use_tta

    logger.info(format_summary(summary, f"ECA-Net ({cfg.data.mode}, attention="
                                        f"{cfg.model.attention}) — pooled OOF, "
                                        f"n = {len(y_true)}"))
    logger.info("  per-fold balanced accuracy: "
                + ", ".join(f"{s*100:.2f}%" for s in fold_scores)
                + f"  (std {np.std(fold_scores)*100:.2f})")

    save_json({"config": cfg.to_dict(), "summary": summary}, out_dir / "results.json")
    np.savez(out_dir / "oof_predictions.npz",
             y_true=y_true, y_prob=y_prob,
             sample_index=np.concatenate(fold_order))

    tag = f"ECA-Net ({cfg.data.mode})"
    plot_confusion_matrix(y_true, y_pred, index.class_names,
                          f"{tag} — pooled out-of-fold", out_dir / "confusion_matrix.png")
    plot_training_curves(histories, tag, out_dir / "curves")
    plot_roc_curves(y_true, y_prob, index.class_names, f"{tag} — ROC",
                    out_dir / "roc.png", cfg.data.mode, index.positive_index)
    plot_precision_recall(y_true, y_prob, index.class_names,
                          f"{tag} — precision-recall", out_dir / "pr.png",
                          cfg.data.mode, index.positive_index)

    total_seconds = sum(h["train_seconds"][0] for h in histories)
    logger.info(f"Finished in {human_time(total_seconds)}. Artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
