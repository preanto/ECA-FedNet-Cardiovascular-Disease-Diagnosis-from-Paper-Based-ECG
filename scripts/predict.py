#!/usr/bin/env python
"""
CardioCare inference engine (Sec. IV-I, Figs. 11-12).

Runs the full deployment path on one or more ECG images: preprocessing
(Table I) -> ECA-Net classification -> predicted class, confidence and an
optional Grad-CAM overlay.  Designed to run on commodity hardware without a
GPU; inference itself takes roughly 4.5 ms per image on a Tesla T4 (Table VI),
with region-of-interest selection dominating end-to-end time.

CardioCare has not been clinically evaluated.  Real-world latency,
out-of-distribution robustness and clinical usability remain unassessed, and no
diagnostic result reported in the manuscript was obtained through it.  This
script is a research prototype, not a medical device.

    python scripts/predict.py --checkpoint model.pth --image scan.png --gradcam
    python scripts/predict.py --checkpoint model.pth --image-dir scans/ --raw
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecanet.config import AugmentationConfig, ModelConfig, PreprocessConfig
from ecanet.data import build_transforms
from ecanet.gradcam import generate_cams, overlay_cam
from ecanet.models import ECANet
from ecanet.preprocessing import IMAGE_SUFFIXES, crop_to_ink, preprocess_image
from ecanet.utils import ensure_dir, setup_logger

DEFAULT_CLASSES = ["Abnormal", "MI", "Normal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CardioCare inference (ECA-Net).")
    parser.add_argument("--checkpoint", required=True, help="Trained ECA-Net weights.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Single image path.")
    source.add_argument("--image-dir", help="Directory of images.")
    parser.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES,
                        help="Class names in the checkpoint's output order.")
    parser.add_argument("--raw", action="store_true",
                        help="Input is an unprocessed scan; apply the Table I pipeline first.")
    parser.add_argument("--header-frac", type=float, default=0.0,
                        help="Header fraction to remove when --raw is used.")
    parser.add_argument("--gradcam", action="store_true", help="Write a Grad-CAM overlay.")
    parser.add_argument("--input-size", type=int, default=ModelConfig.input_size)
    parser.add_argument("--output-dir", default="outputs/predictions")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def collect_images(args) -> list[Path]:
    if args.image:
        return [Path(args.image)]
    root = Path(args.image_dir)
    return sorted(p for p in root.rglob("*")
                  if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    logger = setup_logger(out_dir, filename="predict.log")

    device = torch.device("cpu") if args.cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    model = ECANet(len(args.classes), attention="dual", pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device).eval()
    logger.info(f"ECA-Net loaded on {device} | classes {args.classes}")

    _, eval_tf = build_transforms(args.input_size, AugmentationConfig())
    pre_cfg = PreprocessConfig(output_size=args.input_size)

    paths = collect_images(args)
    if not paths:
        raise SystemExit("No images found.")

    results = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            logger.info(f"  [skip] unreadable: {path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if args.raw:
            rgb = preprocess_image(rgb, pre_cfg, header_frac=args.header_frac)
        else:
            rgb = crop_to_ink(rgb)
        display = cv2.resize(rgb, (args.input_size, args.input_size),
                             interpolation=cv2.INTER_AREA)
        tensor = eval_tf(image=display)["image"].unsqueeze(0).to(device)

        start = time.time()
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor).float(), dim=1)[0].cpu().numpy()
        elapsed_ms = (time.time() - start) * 1000

        best = int(np.argmax(probabilities))
        results.append({"file": path.name, "prediction": args.classes[best],
                        "confidence": float(probabilities[best]),
                        "ms": elapsed_ms})
        detail = "  ".join(f"{name} {p*100:.1f}%"
                           for name, p in zip(args.classes, probabilities))
        logger.info(f"  {path.name}: {args.classes[best]} "
                    f"({probabilities[best]*100:.1f}%) | {detail} | {elapsed_ms:.1f} ms")

        if args.gradcam:
            (cam, class_index, _), = generate_cams(model, tensor.cpu(), device,
                                                   class_indices=[best])
            overlay = overlay_cam(display, cam)
            dst = out_dir / f"{path.stem}_gradcam.png"
            cv2.imwrite(str(dst), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            logger.info(f"    Grad-CAM overlay: {dst}")

    logger.info(f"\nProcessed {len(results)} image(s). "
                "Research prototype — not for clinical use.")


if __name__ == "__main__":
    main()
