import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import os

# ==== CONFIGURATION ====
SRC_DIR = Path('H:\Thesis Experiements\Dataset\Manual Cropping Approach\Dataset')       # <-- your dataset root folder
DST_DIR = Path('Processed Dataset')     # <-- output folder
CANVAS = (3000, 3000)
THRESH = 60
KERNEL = 3
MIN_AREA = 50

# ==== PROCESSING LOOP ====
for fn in tqdm(list(SRC_DIR.rglob('*.*'))):  # includes subfolders
    if not fn.is_file():
        continue

    # Load image
    im = Image.open(fn).convert('RGB')
    arr = np.array(im)
    gray = np.mean(arr, axis=2)
    mask = gray < THRESH
    rgba = np.dstack([arr, mask.astype(np.uint8) * 255])
    ink_only = Image.fromarray(rgba, mode='RGBA')

    # Create white canvas and paste
    canvas = Image.new('RGB', CANVAS, color='white')
    w, h = ink_only.size[:2]
    scale = min(CANVAS[0] / w, CANVAS[1] / h, 1.0)
    if scale < 1:
        new_size = (int(w * scale), int(h * scale))
        ink_only = ink_only.resize(new_size, Image.LANCZOS)
    cx = (CANVAS[0] - ink_only.width) // 2
    cy = (CANVAS[1] - ink_only.height) // 2
    canvas.paste(ink_only, (cx, cy), ink_only)

    # Enhance ECG signal (dilate)
    img_array = np.array(canvas)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    kernel = np.ones((KERNEL, KERNEL), np.uint8)
    dilated = cv2.dilate(inv, kernel, iterations=1)

    # Area filter (remove small specks)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    clean_mask = np.zeros_like(dilated, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA:
            clean_mask[labels == i] = 255

    # Final image (white bg, black signal)
    final_img = 255 - clean_mask
    final_rgb = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
    final_pil = Image.fromarray(final_rgb)

    # Save to DST_DIR (preserve subfolder structure)
    rel_path = fn.relative_to(SRC_DIR)
    out_path = DST_DIR / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_pil.save(out_path.with_suffix('.png'))  # save as PNG
