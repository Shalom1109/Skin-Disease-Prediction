"""Checks the quality-gate thresholds against real images so good photos are not wrongly flagged.

    python scripts/calibrate_quality.py --data-dir dataset --n 400

Prints the percentile spread of each metric on HAM10000 and the share of images the gate would flag, plus how
the gate reacts to artificially blurred / darkened / brightened copies (it should flag nearly all of those).
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))
import quality  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(Path(args.data_dir).rglob("*.jpg"))
    random.Random(args.seed).shuffle(files)
    files = files[: args.n]
    print(f"{len(files)} images from {args.data_dir}")

    metrics = {k: [] for k in ("sharpness", "brightness", "contrast", "clipped_fraction")}
    flagged = 0
    issue_counts = {}
    degraded = {"blur": [0, 0], "dark": [0, 0], "bright": [0, 0]}
    for f in files:
        img = Image.open(f).convert("RGB")
        res = quality.assess_quality(img)
        for k in metrics:
            metrics[k].append(res["metrics"][k])
        if not res["ok"]:
            flagged += 1
            for i in res["issues"]:
                issue_counts[i] = issue_counts.get(i, 0) + 1
        tests = {
            "blur": img.filter(ImageFilter.GaussianBlur(8)),
            "dark": ImageEnhance.Brightness(img).enhance(0.25),
            "bright": ImageEnhance.Brightness(img).enhance(2.6),
        }
        for name, im in tests.items():
            degraded[name][1] += 1
            degraded[name][0] += 0 if quality.assess_quality(im)["ok"] else 1

    print("\nmetric percentiles on real images (1% / 5% / 50% / 95% / 99%):")
    for k, v in metrics.items():
        print(f"  {k:18} " + " / ".join(f"{np.percentile(v, p):8.2f}" for p in (1, 5, 50, 95, 99)))
    print(f"\nflagged as poor quality: {flagged}/{len(files)} = {100 * flagged / len(files):.1f}%  {issue_counts}")
    print("artificially degraded copies caught by the gate:")
    for name, (hit, total) in degraded.items():
        print(f"  {name:7} {hit}/{total} = {100 * hit / total:.0f}%")


if __name__ == "__main__":
    main()
