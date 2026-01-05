import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import albumentations as A

# ========== PARAMETERS ==========
SRC_DIR = "J:\RAW dataset\Raw Aug Compressed"           # input: each class folder within this dir
DST_DIR = "J:\RAW dataset\Raw Augmented"        # output path for (augmented) images
TARGET_IMAGES_PER_CLASS = 3000  # ~10k total for 3 classes
IMG_SIZE = (1000, 1000)           # resize all output here
IMG_FORMAT = ".png"             # change to ".jpg" if needed

# Define only ECG-safe augmentations
ecg_safe_augment = A.Compose([
    A.ShiftScaleRotate(
        shift_limit=0.03,
        scale_limit=0.07,
        rotate_limit=3,    # very small rotation only
        p=0.5,
        border_mode=cv2.BORDER_REFLECT
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.1,
        contrast_limit=0.1,
        p=0.5
    ),
    A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
    A.CLAHE(p=0.2),
    A.GaussianBlur(blur_limit=(3,5), p=0.2)
    # No flips, no elastic, no grid, no hue/saturation
])

def load_image_paths(folder):
    return [os.path.join(folder, fname)
            for fname in os.listdir(folder)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

classes = [c for c in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, c))]
os.makedirs(DST_DIR, exist_ok=True)
for class_name in classes:
    os.makedirs(os.path.join(DST_DIR, class_name), exist_ok=True)

for class_name in classes:
    src_class_dir = os.path.join(SRC_DIR, class_name)
    dst_class_dir = os.path.join(DST_DIR, class_name)
    img_paths = load_image_paths(src_class_dir)
    n_existing = len(img_paths)
    total_needed = TARGET_IMAGES_PER_CLASS
    idx = 0

    # 1. First, copy & resize all originals
    for i, img_path in enumerate(tqdm(img_paths, desc=f"Copying {class_name}")):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        out_path = os.path.join(dst_class_dir, f"orig_{i}{IMG_FORMAT}")
        cv2.imwrite(out_path, img)
        idx += 1

    # 2. Augment until target per-class count reached
    total_aug_needed = total_needed - idx
    aug_idx = 0
    while aug_idx < total_aug_needed:
        img_path = random.choice(img_paths)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        aug_img = ecg_safe_augment(image=img)['image']
        out_path = os.path.join(dst_class_dir, f"aug_{aug_idx}{IMG_FORMAT}")
        cv2.imwrite(out_path, aug_img)
        aug_idx += 1
    print(f"Class {class_name}: total images = {idx + aug_idx}")

# Optional: verify
for class_name in classes:
    count = len(os.listdir(os.path.join(DST_DIR, class_name)))
    print(f"{class_name}: {count} images")
