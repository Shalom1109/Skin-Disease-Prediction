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
