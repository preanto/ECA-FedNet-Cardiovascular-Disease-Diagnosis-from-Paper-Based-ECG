#!/usr/bin/env python
"""
Cross-domain evaluation on an external corpus (Sec. V-A, Table XIII).

The external dataset (ECG Images dataset of Cardiac Patients, Mendeley Data)
contains a fourth class, History of MI, that does not exist in SSMCH-ECG.  The
three-class output layer of the trained ECA-Net is therefore replaced with a
four-class head and adapted on a portion of the external corpus; results are
reported on the remaining held-out samples.

This measures how well the learned features transfer to a new acquisition
setting.  It is not zero-shot generalisation, and the manuscript is explicit on
that point — the History of MI figures in particular must be read with the
class-definition mismatch in mind.

    python scripts/run_external_validation.py \
        --external-root data/external-ecg \
        --checkpoint outputs/ecanet_three_class_dual/ecanet_fold1.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecanet.config import DataConfig, add_common_args, config_from_args
from ecanet.data import (build_index, build_transforms, build_weighted_sampler,
                         make_loader, protocol_b_split)
from ecanet.engine import predict, train_model
from ecanet.losses import build_loss, class_counts_from_targets
from ecanet.metrics import format_summary, summarize
from ecanet.models import ECANet
from ecanet.utils import ensure_dir, save_json, seed_everything, setup_logger
from ecanet.visualization import plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External-corpus evaluation (Table XIII).")
    add_common_args(parser)
    parser.add_argument("--external-root", required=True,
                        help="External dataset root (one folder per class).")
    parser.add_argument("--checkpoint", required=True,
                        help="ECA-Net weights trained on SSMCH-ECG.")
    parser.add_argument("--adapt-fraction", type=float, default=0.8,
                        help="Fraction of the external corpus used to adapt the new head.")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Adapt the head and attention module only (linear probe).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)
    cfg.eval.test_fraction = 1.0 - args.adapt_fraction

    out_dir = ensure_dir(Path(cfg.output_dir) / "external_validation")
    logger = setup_logger(out_dir)
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The external corpus has its own class list; treat every folder as a class.
    external_cfg = DataConfig(root=args.external_root,
                              mode="three_class",
                              num_workers=cfg.data.num_workers,
                              crop_to_ink=cfg.data.crop_to_ink)
    try:
        index = build_index(external_cfg)
    except ValueError:
        # No folder literally named "Normal" — fall back to raw folder names.
        from torchvision import datasets as tvd
        from ecanet.data import DatasetIndex
        base = tvd.ImageFolder(args.external_root)
        index = DatasetIndex(samples=list(base.samples),
                             targets=np.array([l for _, l in base.samples]),
                             class_names=list(base.classes),
                             positive_index=0,
                             raw_classes=list(base.classes))

    logger.info(f"External corpus: {index.size} records | classes {index.class_names} | "
                f"counts {index.counts()}")

    # ---- load the SSMCH-ECG model and replace the output layer ----
    state = torch.load(args.checkpoint, map_location="cpu")
    source_classes = None
    for key, tensor in state.items():
        if key.endswith("classifier.1.weight"):
            source_classes = tensor.shape[0]
    logger.info(f"Checkpoint head covers {source_classes} classes; "
                f"external corpus needs {index.num_classes}.")

    model = ECANet(source_classes or index.num_classes,
                   attention=cfg.model.attention,
                   reduction=cfg.model.reduction_ratio,
                   spatial_kernel=cfg.model.spatial_kernel,
                   dropout=cfg.model.dropout,
                   pretrained=False)
    model.load_state_dict(state)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, index.num_classes)
    logger.info(f"Replaced the classification head: {in_features} -> {index.num_classes}")

    backbone_params, head_params = model.parameter_groups()
    if args.freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
        cfg.train.freeze_backbone_epochs = 0
        logger.info("Backbone frozen: adapting the attention module and head only.")

    # ---- stratified adapt / held-out split of the external corpus ----
    adapt_idx, holdout_idx = protocol_b_split(index, cfg.eval, cfg.seed)
    logger.info(f"Adaptation split: {len(adapt_idx)} adapt | {len(holdout_idx)} held out")

    train_tf, eval_tf = build_transforms(cfg.model.input_size, cfg.augment)
    sampler = build_weighted_sampler(index.targets, adapt_idx, index.num_classes, cfg.seed)
    adapt_loader = make_loader(index, adapt_idx, train_tf, cfg.train.batch_size,
                               True, external_cfg, sampler=sampler)
    holdout_loader = make_loader(index, holdout_idx, eval_tf, cfg.train.batch_size,
                                 False, external_cfg)

    counts = class_counts_from_targets(index.targets, adapt_idx, index.num_classes)
    criterion = build_loss(cfg.train.loss, counts, device,
                           beta=cfg.train.cb_beta,
                           gamma=cfg.train.focal_gamma,
                           focal_label_smoothing=cfg.train.focal_label_smoothing,
                           ce_label_smoothing=cfg.train.ce_label_smoothing)

    model, _ = train_model(model, adapt_loader, holdout_loader, criterion, device,
                           cfg.train, cfg.eval,
                           head_params=head_params,
                           backbone_params=None if args.freeze_backbone else backbone_params,
                           log_fn=logger.info, tag="external")

    y_true, y_prob = predict(model, holdout_loader, device,
                             use_tta=cfg.eval.use_tta, tta_angles=cfg.eval.tta_angles)
    y_pred = y_prob.argmax(1)
    summary = summarize(y_true, y_pred, y_prob, index.class_names,
                        index.positive_index, "three_class",
                        cfg.eval.ci_method, cfg.eval.ci_level,
                        cfg.eval.n_bootstrap, cfg.seed)
    summary["protocol"] = ("External corpus; head replaced and adapted on "
                           f"{len(adapt_idx)} records, evaluated on {len(holdout_idx)}.")

    logger.info(format_summary(summary, "External corpus — held-out evaluation"))
    logger.info("Class definitions differ between corpora (History of MI has no "
                "SSMCH-ECG counterpart); interpret per-class figures accordingly.")

    plot_confusion_matrix(y_true, y_pred, index.class_names,
                          "Generalisation test — external corpus",
                          out_dir / "confusion_matrix.png")
    torch.save(model.state_dict(), out_dir / "ecanet_external.pth")
    save_json({"config": cfg.to_dict(), "summary": summary}, out_dir / "results.json")
    logger.info(f"Artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
