"""Photo-quality check: tells the user, before or alongside the analysis, whether the photo is likely good enough.

Cheap, model-free heuristics (numpy + Pillow only). Thresholds were set from the distribution of HAM10000
images so that good dermoscopy photos are not flagged (see tests and scripts/calibrate_quality.py).
"""
import numpy as np
from PIL import Image, ImageFilter

# thresholds (luminance 0-255, sharpness = variance of a Laplacian on a 512-px grey image, centre crop).
# Real HAM10000 photos score >= 14 at the 1st percentile; Gaussian-blurred copies score about 2, so 8 separates them.
MIN_SIDE = 224
MIN_SHARPNESS = 8.0      # measured on the CENTRAL half of the photo (ignores dermoscope vignettes/borders)
MIN_BRIGHTNESS = 45.0
MAX_BRIGHTNESS = 215.0
MIN_CONTRAST = 6.0
MAX_CLIPPED = 0.20       # fraction of pixels that are nearly pure white or black

_LAPLACIAN = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128)

TIPS = {
    "low_resolution": "The photo is small. Move a little closer or use the camera's full resolution.",
    "blurry": "The photo looks blurry. Hold the camera steady, tap the spot to focus, and retake it.",
    "too_dark": "The photo is too dark. Use bright, even, natural light and avoid shadows.",
    "too_bright": "The photo is overexposed. Avoid direct flash or strong glare on the skin.",
    "low_contrast": "The photo looks flat or washed out. Check the lighting and that the skin fills the frame.",
    "clipped": "Large areas are pure white or black (glare or shadow). Reposition the light and retake.",
}


def assess_quality(image: Image.Image) -> dict:
    rgb = image.convert("RGB")
    w, h = rgb.size
    gray = rgb.convert("L")
    # normalise scale so sharpness is comparable between a 600-px and a 4000-px photo
    scale = 512 / max(w, h)
    small = gray.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    lap_full = np.asarray(small.filter(_LAPLACIAN), dtype=np.float64) - 128.0
    arr = np.asarray(small, dtype=np.float64)
    # Sharpness only on the central half: black vignettes / hard borders would otherwise dominate the measure.
    hh, ww = lap_full.shape
    lap = lap_full[hh // 4: hh - hh // 4, ww // 4: ww - ww // 4]

    metrics = {
        "width": w, "height": h,
        "sharpness": float(lap.var()),
        "brightness": float(arr.mean()),
        "contrast": float(arr.std()),
        "clipped_fraction": float(((arr > 250) | (arr < 5)).mean()),
    }
    issues = []
    if min(w, h) < MIN_SIDE:
        issues.append("low_resolution")
    if metrics["sharpness"] < MIN_SHARPNESS:
        issues.append("blurry")
    if metrics["brightness"] < MIN_BRIGHTNESS:
        issues.append("too_dark")
    elif metrics["brightness"] > MAX_BRIGHTNESS:
        issues.append("too_bright")
    if metrics["contrast"] < MIN_CONTRAST:
        issues.append("low_contrast")
    if metrics["clipped_fraction"] > MAX_CLIPPED:
        issues.append("clipped")

    return {
        "ok": not issues,
        "issues": issues,
        "tips": [TIPS[i] for i in issues],
        "metrics": {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()},
    }
