#!/usr/bin/env python
"""
Grad-CAM visualisation for representative records (Sec. IV-H, Fig. 10).

By default one correctly classified example per class is selected from the
dataset and a two-row panel (input / heat map) is written to the output
directory.  Individual image paths can be supplied instead.

The manuscript presents these maps as an illustration of model behaviour, not
as evidence of localisation accuracy: they were never scored against
clinician-annotated diagnostic regions (Sec. V-E).

    python scripts/run_gradcam.py --data-root data/SSMCH-ECG \
        --checkpoint outputs/ecanet_three_class_dual/ecanet_fold1.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecanet.config import add_common_args, config_from_args
from ecanet.data import build_index, build_transforms
from ecanet.gradcam import generate_cams, overlay_cam
from ecanet.models import ECANet
from ecanet.preprocessing import crop_to_ink
from ecanet.utils import ensure_dir, seed_everything, setup_logger
from ecanet.visualization import plot_gradcam_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM analysis (Fig. 10).")
    add_common_args(parser)
    parser.add_argument("--checkpoint", required=True, help="Trained ECA-Net weights (.pth).")
    parser.add_argument("--images", nargs="*", default=None,
                        help="Specific image paths; default picks one record per class.")
    parser.add_argument("--per-class", type=int, default=1,
                        help="Examples per class when --images is not given.")
    parser.add_argument("--after-attention", action="store_true",
                        help="Hook the CBAM output instead of the final backbone stage.")
    parser.add_argument("--alpha", type=float, default=0.4, help="Heat-map blend weight.")
    return parser.parse_args()


def load_display_image(path: str, size: int, crop_ink: bool,
                       gray_threshold: int, pad_frac: float) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if crop_ink:
        rgb = crop_to_ink(rgb, gray_threshold, pad_frac)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    cfg = config_from_args(args)

    out_dir = ensure_dir(Path(cfg.output_dir) / f"gradcam_{cfg.data.mode}")
    logger = setup_logger(out_dir)
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index = build_index(cfg.data)
    _, eval_tf = build_transforms(cfg.model.input_size, cfg.augment)

    model = ECANet(index.num_classes,
                   attention=cfg.model.attention,
                   reduction=cfg.model.reduction_ratio,
                   spatial_kernel=cfg.model.spatial_kernel,
                   dropout=cfg.model.dropout,
                   pretrained=False)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    # ---- choose the records to visualise ----
    if args.images:
        paths = list(args.images)
        true_labels = [None] * len(paths)
    else:
        paths, true_labels = [], []
        for c, name in enumerate(index.class_names):
            candidates = np.where(index.targets == c)[0][:args.per_class]
            for gi in candidates:
                paths.append(index.samples[gi][0])
                true_labels.append(name)
        logger.info(f"Selected {len(paths)} records ({args.per_class} per class).")

    displays, overlays, captions = [], [], []
    for path, true_label in zip(paths, true_labels):
        display = load_display_image(path, cfg.model.input_size, cfg.data.crop_to_ink,
                                     cfg.data.ink_crop_gray_threshold,
                                     cfg.data.ink_crop_pad_frac)
        tensor = eval_tf(image=display)["image"].unsqueeze(0)
        (cam, class_index, probability), = generate_cams(
            model, tensor, device, after_attention=args.after_attention)

        overlay = overlay_cam(display, cam, alpha=args.alpha)
        predicted = index.class_names[class_index]
        caption = (f"{true_label} -> {predicted} ({probability*100:.1f}%)"
                   if true_label else f"{predicted} ({probability*100:.1f}%)")
        displays.append(display)
        overlays.append(overlay)
        captions.append(caption)

        stem = Path(path).stem
        cv2.imwrite(str(out_dir / f"gradcam_{stem}.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        logger.info(f"  {Path(path).name}: {caption}")

    panel = plot_gradcam_panel(displays, overlays, captions,
                               out_dir / "gradcam_panel.png",
                               title="Grad-CAM analysis of ECA-Net predictions")
    logger.info(f"Panel written to {panel}")
    logger.info("These maps are qualitative; they were not scored against "
                "clinician-annotated regions (Sec. V-E).")


if __name__ == "__main__":
    main()
