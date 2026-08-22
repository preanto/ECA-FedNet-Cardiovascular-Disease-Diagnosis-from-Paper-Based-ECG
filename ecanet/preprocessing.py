"""
Offline preprocessing pipeline for paper-based ECG scans.

Faithful implementation of Table I of the manuscript
(PREPROCESSING PIPELINE PARAMETERS), Sec. III-C:

  1. Manual ROI cropping        removes the header carrying patient identifiers
  2. Grayscale conversion       ITU-R BT.601 luma
  3. Intensity thresholding     T = 60
  4. Ink-mask extraction        binary RGBA ink-only layer
  5. Canvas normalisation       3000 x 3000 px, white, aspect ratio preserved
  6. Grayscale inversion        aligns polarity with ImageNet statistics
  7. Morphological dilation     3 x 3 kernel, 1 iteration
  8. Connected-component filter minimum area 50 px
  9. Lossless export            PNG
 10. Ink-bounding-box crop      grey threshold < 245, 3 % padding
 11. Resize                     300 x 300

Step 1 is semi-automatic in the paper (Sec. V-E lists this as a limitation).
Here it is exposed as an optional fractional crop so the pipeline can be run
end-to-end in batch; supply --header-frac 0.0 for images whose header has
already been removed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .config import PreprocessConfig

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Individual steps (kept separate so Fig. 3 can be regenerated stage by stage)
# ---------------------------------------------------------------------------
def crop_header(img: np.ndarray, header_frac: float = 0.0) -> np.ndarray:
    """Step 1 — remove the top `header_frac` of the image (patient identifiers)."""
    if header_frac <= 0:
        return img
    y0 = int(round(img.shape[0] * float(header_frac)))
    return img[y0:, :]


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Step 2 — ITU-R BT.601 luma (the weighting used by cv2.COLOR_RGB2GRAY)."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def extract_ink_mask(gray: np.ndarray, threshold: int = 60) -> np.ndarray:
    """Steps 3-4 — pixels darker than T belong to the trace; returns a uint8 mask."""
    mask = (gray < int(threshold)).astype(np.uint8) * 255
    return mask


def fit_to_canvas(mask: np.ndarray, canvas_size: int = 3000) -> np.ndarray:
    """Step 5 — place the ink layer on a square white canvas, aspect ratio kept.

    The mask is white-on-black (ink = 255); the canvas is built in the same
    convention and inverted in step 6.
    """
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((canvas_size, canvas_size), np.uint8)
    scale = min(canvas_size / w, canvas_size / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((canvas_size, canvas_size), np.uint8)
    y0 = (canvas_size - new_h) // 2
    x0 = (canvas_size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def invert(mask: np.ndarray) -> np.ndarray:
    """Step 6 — black trace on a white background."""
    return cv2.bitwise_not(mask)


def dilate_trace(mask: np.ndarray, kernel: int = 3, iterations: int = 1) -> np.ndarray:
    """Step 7 — reconnect discontinuities caused by ink fading.

    Operates on the white-on-black ink mask.
    """
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    return cv2.dilate(mask, k, iterations=iterations)


def filter_small_components(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """Step 8 — drop connected components smaller than `min_area` pixels."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    keep = np.zeros_like(mask)
    for i in range(1, num):  # 0 is background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep


def crop_to_ink(img: np.ndarray, gray_threshold: int = 245, pad_frac: float = 0.03) -> np.ndarray:
    """Step 10 — crop to the ink bounding box with fractional padding.

    Accepts RGB or grayscale.  Returns the input unchanged when fewer than 50
    dark pixels are present (degenerate scan), mirroring the released code.
    """
    gray = to_grayscale(img)
    coords = np.argwhere(gray < gray_threshold)
    if coords.shape[0] < 50:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    h, w = gray.shape[:2]
    pad_y = int((y1 - y0) * pad_frac)
    pad_x = int((x1 - x0) * pad_frac)
    return img[max(0, y0 - pad_y):min(h, y1 + pad_y),
               max(0, x0 - pad_x):min(w, x1 + pad_x)]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def preprocess_image(img_rgb: np.ndarray,
                     cfg: Optional[PreprocessConfig] = None,
                     header_frac: float = 0.0,
                     return_stages: bool = False):
    """Run Table I steps 1-11 on a single RGB image.

    Returns the final 300 x 300 RGB image, or (image, stages) when
    `return_stages` is True — the stage dictionary reproduces Fig. 3.
    """
    cfg = cfg or PreprocessConfig()
    stages: Dict[str, np.ndarray] = {"a_original": img_rgb}

    cropped = crop_header(img_rgb, header_frac)
    stages["b_cropped"] = cropped

    gray = to_grayscale(cropped)
    stages["c_grayscale"] = gray

    ink = extract_ink_mask(gray, cfg.ink_threshold)
    stages["d_threshold"] = ink
    stages["e_ink_layer"] = ink

    canvas = fit_to_canvas(ink, cfg.canvas_size)
    stages["f_canvas"] = canvas

    dilated = dilate_trace(canvas, cfg.dilation_kernel, cfg.dilation_iterations)
    stages["h_dilated"] = dilated

    cleaned = filter_small_components(dilated, cfg.min_component_area)
    stages["i_components"] = cleaned

    inverted = invert(cleaned)                       # step 6, applied after cleaning
    stages["g_inverted"] = inverted

    bounded = crop_to_ink(inverted, cfg.bbox_gray_threshold, cfg.bbox_pad_frac)
    resized = cv2.resize(bounded, (cfg.output_size, cfg.output_size),
                         interpolation=cv2.INTER_AREA)
    final = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    stages["j_final"] = final

    return (final, stages) if return_stages else final


def preprocess_directory(input_root: str | Path,
                         output_root: str | Path,
                         cfg: Optional[PreprocessConfig] = None,
                         header_frac: float = 0.0,
                         verbose: bool = True) -> Tuple[int, int]:
    """Apply the pipeline to every image under `input_root`, preserving the
    class-folder structure expected by torchvision's ImageFolder.

    Returns (processed, failed).
    """
    cfg = cfg or PreprocessConfig()
    input_root, output_root = Path(input_root), Path(output_root)
    processed = failed = 0

    for src in sorted(input_root.rglob("*")):
        if src.suffix.lower() not in IMAGE_SUFFIXES or not src.is_file():
            continue
        rel = src.relative_to(input_root)
        dst = (output_root / rel).with_suffix("." + cfg.output_format)
        dst.parent.mkdir(parents=True, exist_ok=True)
        raw = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if raw is None:
            failed += 1
            if verbose:
                print(f"  [skip] unreadable: {src}")
            continue
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        try:
            out = preprocess_image(rgb, cfg, header_frac=header_frac)
        except Exception as exc:                      # noqa: BLE001 - report and continue
            failed += 1
            if verbose:
                print(f"  [fail] {src}: {exc}")
            continue
        cv2.imwrite(str(dst), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        processed += 1
        if verbose and processed % 100 == 0:
            print(f"  processed {processed} images ...")

    return processed, failed


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ECA-Net preprocessing pipeline (Table I).")
    p.add_argument("--input", required=True, help="Raw ECG image root (class subfolders).")
    p.add_argument("--output", required=True, help="Destination for preprocessed images.")
    p.add_argument("--threshold", type=int, default=PreprocessConfig.ink_threshold,
                   help="Ink threshold T (Table I, step 3).")
    p.add_argument("--canvas-size", type=int, default=PreprocessConfig.canvas_size)
    p.add_argument("--min-area", type=int, default=PreprocessConfig.min_component_area)
    p.add_argument("--output-size", type=int, default=PreprocessConfig.output_size)
    p.add_argument("--header-frac", type=float, default=0.0,
                   help="Fraction of image height to remove as header (step 1).")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg = PreprocessConfig(ink_threshold=args.threshold,
                           canvas_size=args.canvas_size,
                           min_component_area=args.min_area,
                           output_size=args.output_size)
    print(f"Preprocessing {args.input} -> {args.output} (T={cfg.ink_threshold})")
    done, failed = preprocess_directory(args.input, args.output, cfg,
                                        header_frac=args.header_frac)
    print(f"Done. {done} images written, {failed} failed.")


if __name__ == "__main__":
    main()
