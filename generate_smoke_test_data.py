"""
Generate a tiny SYNTHETIC dataset for smoke-testing the diabetic
retinopathy pipeline.

This does NOT create real fundus images and is NOT meant to produce a
medically meaningful model - it exists purely so `diabetic_retinopathy_
detection.py` can be run end-to-end (data loading -> preprocessing ->
model build -> training -> evaluation -> Grad-CAM) to confirm there are
no bugs, without requiring a real DR dataset or internet access.

Replace this with a real dataset before drawing any conclusions - see
README.md for download instructions.

Usage:
    python generate_smoke_test_data.py --out_dir data --n_per_class 12
"""

import os
import argparse
import numpy as np
from PIL import Image, ImageDraw

CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]


def make_synthetic_fundus(severity_idx, size=224, seed=0):
    """Draws a circular 'fundus-like' image with a number of random
    blobs proportional to severity_idx, so the classes are at least
    weakly separable - good enough to exercise the training loop."""
    rng = np.random.RandomState(seed)
    img = Image.new("RGB", (size, size), (10, 10, 10))
    draw = ImageDraw.Draw(img)

    # circular fundus background
    base_color = (rng.randint(150, 200), rng.randint(60, 100), rng.randint(40, 70))
    draw.ellipse([4, 4, size - 4, size - 4], fill=base_color)

    # optic disc
    cx, cy = size // 2 + rng.randint(-20, 20), size // 2 + rng.randint(-20, 20)
    draw.ellipse([cx - 15, cy - 15, cx + 15, cy + 15],
                 fill=(230, 200, 150))

    # "lesions" - more + darker as severity increases
    n_lesions = severity_idx * rng.randint(3, 6)
    for _ in range(n_lesions):
        x, y = rng.randint(20, size - 20), rng.randint(20, size - 20)
        r = rng.randint(2, 5 + severity_idx * 2)
        color = (rng.randint(80, 120), 10, 10)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--n_per_class", type=int, default=12)
    args = parser.parse_args()

    seed = 0
    for split, n in [("train", args.n_per_class), ("val", max(3, args.n_per_class // 4))]:
        for idx, cls in enumerate(CLASS_NAMES):
            cls_dir = os.path.join(args.out_dir, split, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(n):
                img = make_synthetic_fundus(idx, seed=seed)
                seed += 1
                img.save(os.path.join(cls_dir, f"{cls}_{i}.png"))
        print(f"{split}: {n} synthetic images x {len(CLASS_NAMES)} classes")

    print(f"\nSynthetic smoke-test dataset written to '{args.out_dir}/'")
    print("Reminder: this is NOT real fundus data - for code-verification only.")


if __name__ == "__main__":
    main()
