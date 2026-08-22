#!/usr/bin/env python
"""
Transfer-learning comparison under Protocol B (Table X).

Every architecture is trained on the same fixed stratified 80/20 split with the
same augmentation, optimiser, schedule and test-time augmentation, then scored
on the identical 200-record held-out set.  Models are ranked by balanced
accuracy, as in the manuscript.

    python scripts/run_benchmark.py --data-root data/SSMCH-ECG
    python scripts/run_benchmark.py --data-root data/SSMCH-ECG --include-extra
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecanet.config import BASELINE_MODELS, EXTRA_MODELS, add_common_args, config_from_args
from ecanet.data import (build_index, build_transforms, build_weighted_sampler,
                         make_loader, protocol_b_split)
from ecanet.engine import measure_complexity, predict, train_model
from ecanet.losses import build_loss, class_counts_from_targets
from ecanet.metrics import format_summary, summarize
from ecanet.models import build_model
from ecanet.utils import ensure_dir, save_json, seed_everything, setup_logger
from ecanet.visualization import plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline benchmark (Table X).")
    add_common_args(parser)
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of model names to run.")
    parser.add_argument("--include-extra", action="store_true",
                        help="Also run the screened architectures not reported in the paper.")
    parser.add_argument("--skip-complexity", action="store_true",
                        help="Skip the parameter/GFLOPs/latency measurement.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)

    out_dir = ensure_dir(Path(cfg.output_dir) / f"benchmark_{cfg.data.mode}")
    logger = setup_logger(out_dir)
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    index = build_index(cfg.data)
    logger.info(f"Loaded {index.size} records | classes {index.class_names} | "
                f"counts {index.counts()}")

    model_list = list(BASELINE_MODELS)
    if args.include_extra:
        model_list += EXTRA_MODELS
    if args.models:
        wanted = set(args.models)
        model_list = [m for m in model_list if m[0] in wanted]
        missing = wanted - {m[0] for m in model_list}
        if missing:
            raise SystemExit(f"Unknown model name(s): {sorted(missing)}")

    train_idx, test_idx = protocol_b_split(index, cfg.eval, cfg.seed)
    logger.info(f"Protocol B split: train {len(train_idx)} | test {len(test_idx)}")

    rows = []
    all_summaries = {}

    for name, input_size in model_list:
        logger.info("=" * 70)
        logger.info(f"MODEL: {name} (input {input_size} x {input_size})")
        seed_everything(cfg.seed)

        cfg.model.input_size = input_size
        train_tf, eval_tf = build_transforms(input_size, cfg.augment)
        sampler = (build_weighted_sampler(index.targets, train_idx, index.num_classes,
                                          cfg.seed)
                   if cfg.train.use_weighted_sampler else None)
        train_loader = make_loader(index, train_idx, train_tf, cfg.train.batch_size,
                                   True, cfg.data, sampler=sampler)
        test_loader = make_loader(index, test_idx, eval_tf, cfg.train.batch_size,
                                  False, cfg.data)

        model, head_params = build_model(name, index.num_classes, cfg.model)
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

        counts = class_counts_from_targets(index.targets, train_idx, index.num_classes)
        criterion = build_loss(cfg.train.loss, counts, device,
                               beta=cfg.train.cb_beta,
                               gamma=cfg.train.focal_gamma,
                               focal_label_smoothing=cfg.train.focal_label_smoothing,
                               ce_label_smoothing=cfg.train.ce_label_smoothing)

        model, _ = train_model(model, train_loader, test_loader, criterion, device,
                               cfg.train, cfg.eval,
                               head_params=head_params,
                               backbone_params=backbone_params,
                               log_fn=logger.info, tag=name)

        y_true, y_prob = predict(model, test_loader, device,
                                 use_tta=cfg.eval.use_tta,
                                 tta_angles=cfg.eval.tta_angles)
        y_pred = y_prob.argmax(1)
        summary = summarize(y_true, y_pred, y_prob, index.class_names,
                            index.positive_index, cfg.data.mode,
                            cfg.eval.ci_method, cfg.eval.ci_level,
                            cfg.eval.n_bootstrap, cfg.seed)
        all_summaries[name] = summary
        logger.info(format_summary(summary, f"{name} — held-out test"))

        complexity = {"params_m": float("nan"), "gflops": float("nan"),
                      "ms_per_image": float("nan")}
        if not args.skip_complexity:
            fresh, _ = build_model(name, index.num_classes, cfg.model)
            complexity = measure_complexity(fresh, input_size, device, cfg.eval)
            del fresh

        bal_lo, bal_hi = summary["balanced_accuracy_ci"]
        row = {
            "Model": name,
            "Input": input_size,
            "Acc (%)": round(summary["accuracy"] * 100, 2),
            "Bal. Acc (%)": round(summary["balanced_accuracy"] * 100, 2),
            "Bal. Acc 95% CI": f"{bal_lo*100:.2f}-{bal_hi*100:.2f}",
            "Macro-F1": round(summary["macro_f1"], 4),
            f"{index.class_names[index.positive_index]} recall (%)":
                round(summary["positive_sensitivity"] * 100, 2),
            "MCC": round(summary["mcc"], 4),
            "AUC": round(summary.get("auc", float("nan")), 4),
            "Params (M)": round(complexity["params_m"], 2),
            "GFLOPs": (None if np.isnan(complexity["gflops"])
                       else round(complexity["gflops"], 2)),
            "Latency (ms/img)": round(complexity["ms_per_image"], 1),
        }
        # Per-class precision / recall / F1 as laid out in Table X
        for cls in summary["per_class"]:
            row[f"{cls['Class']} P"] = round(cls["Precision"] * 100, 2)
            row[f"{cls['Class']} R"] = round(cls["Recall"] * 100, 2)
            row[f"{cls['Class']} F1"] = round(cls["F1"] * 100, 2)
        rows.append(row)

        plot_confusion_matrix(y_true, y_pred, index.class_names,
                              f"{name} ({cfg.data.mode})",
                              out_dir / f"cm_{name}.png")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values("Bal. Acc (%)", ascending=False).reset_index(drop=True)
    df.to_csv(out_dir / "comparison_table.csv", index=False)

    core = ["Model", "Input", "Acc (%)", "Bal. Acc (%)", "Bal. Acc 95% CI",
            "Macro-F1", "MCC", "AUC", "Params (M)", "GFLOPs", "Latency (ms/img)"]
    logger.info("\n" + "=" * 78)
    logger.info(f"COMPARISON TABLE ({cfg.data.mode}) — ranked by balanced accuracy")
    logger.info("=" * 78 + "\n" + df[core].to_string(index=False))

    save_json({"config": cfg.to_dict(), "summaries": all_summaries},
              out_dir / "results.json")
    logger.info(f"Artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
