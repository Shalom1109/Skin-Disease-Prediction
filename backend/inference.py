"""Pre-processing and decision logic for a single image.

The old `dynamic_calibrate_logits` (hand-tuned logit offsets fitted to a few
sample pictures) is gone. What replaces it is principled and data-driven:

* temperature scaling      - one scalar fitted on the validation split
* melanoma escalation rule - a probability threshold chosen on the validation
                             split for a target sensitivity
* low-confidence rejection - below `min_confidence` the image is reported as
                             uncertain

All three values come from backend/calibration.json, written by evaluate.py.
"""
import numpy as np
import torch
import torchvision.transforms as transforms

IMG_SIZE = 384
TRANSFORM = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess(image):
    return TRANSFORM(image).unsqueeze(0)


def predict_probs(model, tensor, temperature=1.0):
    try:
        device = next(model.parameters()).device
    except StopIteration:                      # parameter-less stand-in models (tests)
        device = torch.device("cpu")
    # Mixed precision on CUDA, exactly like evaluate.py, so the fitted temperature and
    # melanoma threshold in calibration.json were measured on the same kind of logits.
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits = model(tensor.to(device))[0].float().double().cpu().numpy()
    z = logits / float(temperature)
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def decide(probs, class_names, cfg, melanoma="Melanoma",
           escalated_label="Melanoma (High-Priority Escalation)",
           uncertain_label="Uncertain / Non-Skin Image"):
    probs = np.asarray(probs, dtype=float)
    mel_idx = class_names.index(melanoma)
    top_idx = int(probs.argmax())
    top_conf = float(probs[top_idx])
    mel_p = float(probs[mel_idx])

    all_probs = {name: round(float(p) * 100, 1) for name, p in zip(class_names, probs)}
    base = {
        "top_prediction": class_names[top_idx],
        "top_confidence": round(top_conf * 100, 1),
        "melanoma_probability": round(mel_p * 100, 1),
        "probabilities": all_probs,
        "escalated": False,
        "uncertain": False,
        "note": "",
    }

    if top_idx != mel_idx and mel_p >= cfg["melanoma_threshold"]:
        return {**base,
                "prediction": escalated_label,
                "class_name": melanoma,
                "confidence": round(mel_p * 100, 1),
                "escalated": True,
                "note": (f"Flagged for safety: melanoma probability is {mel_p * 100:.1f}% "
                         f"(escalation threshold {cfg['melanoma_threshold'] * 100:.1f}%), although the "
                         f"most likely class was {class_names[top_idx]} ({top_conf * 100:.1f}%). "
                         "This is a screening flag, not a diagnosis - see a dermatologist.")}

    if top_conf < cfg["min_confidence"]:
        return {**base,
                "prediction": uncertain_label,
                "class_name": None,
                "confidence": round(top_conf * 100, 1),
                "uncertain": True,
                "note": "The model is not confident about this image."}

    return {**base,
            "prediction": class_names[top_idx],
            "class_name": class_names[top_idx],
            "confidence": round(top_conf * 100, 1)}


# ---------------------------------------------------------------------------
# Referral guidance: what the person should DO, in plain language
# ---------------------------------------------------------------------------
STANDING_ADVICE = ("This tool can miss skin cancers. If a spot is new, changing, growing, bleeding, itching or "
                   "simply worries you, see a doctor whatever this result says.")


def referral(result, quality=None):
    """Turn a decision (from `decide`) into a referral level. Deliberately conservative:
    anything suggestive of melanoma or basal cell carcinoma is 'priority'."""
    cls = result.get("class_name")
    if result.get("escalated") or cls in ("Melanoma", "Basal Cell Carcinoma"):
        out = {"level": "priority", "title": "See a dermatologist promptly",
               "timeframe": "Within 1-2 weeks",
               "message": "This photo has features the model associates with a skin cancer. That is not a diagnosis, "
                          "but it is worth having examined by a dermatologist soon."}
    elif cls == "Actinic Keratoses":
        out = {"level": "soon", "title": "Book a dermatology review",
               "timeframe": "Within a few weeks",
               "message": "This resembles a sun-damage spot that can occasionally progress. A doctor should look at it."}
    elif result.get("uncertain") or cls is None:
        out = {"level": "inconclusive", "title": "Result not reliable",
               "timeframe": "Retake the photo",
               "message": "The model could not read this photo with confidence. Retake it in good light, or see a "
                          "doctor if you are concerned."}
    else:
        out = {"level": "routine", "title": "Likely harmless - keep an eye on it",
               "timeframe": "Re-check in about 3 months",
               "message": "This resembles a common non-cancerous spot. Photograph it again in a few months and see a "
                          "doctor if it changes in size, shape or colour."}
    if quality is not None and not quality.get("ok", True) and out["level"] in ("routine", "inconclusive"):
        out["message"] += " The photo quality is poor, so retaking it would give a more reliable result."
    out["advice"] = STANDING_ADVICE
    return out
