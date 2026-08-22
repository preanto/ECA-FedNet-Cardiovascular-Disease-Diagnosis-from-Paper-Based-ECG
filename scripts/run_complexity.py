#!/usr/bin/env python
"""
Model size, computational cost and memory footprint.

Reproduces Table IV (layer-wise parameters), Table VI (params / GFLOPs /
latency for every evaluated architecture), Table VII (ECA-Net parameter
distribution) and Table VIII (forward-pass memory footprint).

Latency follows the protocol of Sec. IV-B: batch size 8, four warm-up
iterations, then 20 timed forward passes.  The manuscript's figures were
measured on a Tesla T4; absolute numbers will differ on other hardware, but the
relative ordering should hold.

    python scripts/run_complexity.py
    python scripts/run_complexity.py --num-classes 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecanet.config import BASELINE_MODELS, ModelConfig
from ecanet.engine import measure_complexity, measure_memory_footprint
from ecanet.models import ECANet, build_model
from ecanet.utils import ensure_dir, save_json, seed_everything, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complexity tables (IV, VI, VII, VIII).")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU measurement.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Skip downloading ImageNet weights (parameter counts are identical).")
    return parser.parse_args()


def _fmt_bytes(n: int) -> str:
    """Decimal units, matching Table VIII (1536 x 10 x 10 x 4 B = 614 KB)."""
    if n >= 1_000_000:
        return f"{n / 1e6:.1f} MB"
    if n >= 1_000:
        return f"{n / 1e3:.0f} KB"
    return f"{n} B"


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.output_dir) / "complexity")
    logger = setup_logger(out_dir)
    seed_everything(args.seed)

    device = torch.device("cpu") if args.cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type != "cuda":
        logger.info("Latency measured on CPU; the manuscript reports Tesla T4 timings.")

    model_cfg = ModelConfig(pretrained=not args.no_pretrained)

    # ---- Table VII: ECA-Net parameter distribution ----
    eca = ECANet(args.num_classes,
                 attention=model_cfg.attention,
                 reduction=model_cfg.reduction_ratio,
                 spatial_kernel=model_cfg.spatial_kernel,
                 dropout=model_cfg.dropout,
                 pretrained=model_cfg.pretrained)
    breakdown = eca.parameter_breakdown()
    total = breakdown["Total"]
    logger.info("\n" + "=" * 70)
    logger.info("TABLE VII — ECA-Net parameter distribution")
    logger.info("=" * 70)
    for key, value in breakdown.items():
        share = "" if key == "Total" else f"  ({value / total * 100:.4f} %)"
        logger.info(f"  {key:<28}{value:>12,d}{share}")
    attention_total = breakdown["Channel Attention"] + breakdown["Spatial Attention"]
    logger.info(f"  {'Attention module (CA + SA)':<28}{attention_total:>12,d}"
                f"  ({attention_total / total * 100:.2f} %)")

    # ---- Table VIII: memory footprint ----
    memory_rows = measure_memory_footprint(eca, model_cfg.input_size, device, batch=1)
    logger.info("\n" + "=" * 70)
    logger.info("TABLE VIII — forward-pass memory footprint (FP32, B = 1)")
    logger.info("=" * 70)
    for row in memory_rows:
        logger.info(f"  {row['Stage']:<22}{str(row['Shape']):<24}"
                    f"{_fmt_bytes(row['Bytes_FP32']):>10}")
    del eca

    # ---- Table VI: complexity of every evaluated architecture ----
    rows = []
    for name, input_size in BASELINE_MODELS:
        model, _ = build_model(name, args.num_classes, model_cfg)
        stats = measure_complexity(model, input_size, device)
        rows.append({
            "Model": name,
            "Input": f"{input_size}^2",
            "Params (M)": round(stats["params_m"], 2),
            "GFLOPs": (None if np.isnan(stats["gflops"]) else round(stats["gflops"], 2)),
            "Latency (ms/img)": round(stats["ms_per_image"], 1),
        })
        logger.info(f"  {name:<22} {rows[-1]['Params (M)']:>7} M  "
                    f"{rows[-1]['GFLOPs']} GFLOPs  "
                    f"{rows[-1]['Latency (ms/img)']} ms/img")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values("Params (M)").reset_index(drop=True)
    df.to_csv(out_dir / "complexity_table.csv", index=False)
    logger.info("\n" + "=" * 70)
    logger.info("TABLE VI — computational cost of the evaluated architectures")
    logger.info("=" * 70 + "\n" + df.to_string(index=False))

    save_json({"parameter_breakdown": breakdown,
               "memory_footprint": memory_rows,
               "complexity": rows,
               "device": str(device)},
              out_dir / "complexity.json")
    logger.info(f"\nArtifacts written to {out_dir}")


if __name__ == "__main__":
    main()
