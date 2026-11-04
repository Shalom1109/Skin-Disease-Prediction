"""Evaluation + calibration utilities (numpy/torch only, no web dependencies)."""
import math

import numpy as np
import torch
import torch.nn.functional as F


def softmax(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_temperature(logits, labels, max_iter=200):
    """Temperature scaling (Guo et al. 2017): the single scalar T > 0 that
    minimises validation NLL. It changes confidence, never the predicted class."""
    logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits_t / log_t.exp(), labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def nll(probs, labels):
    probs = np.asarray(probs)
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0))))


def expected_calibration_error(probs, labels, n_bins=15):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def pick_sensitivity_threshold(p_pos, is_pos, target_sensitivity=0.90):
    """Largest threshold t such that the rule `p_pos >= t` still catches at least
    `target_sensitivity` of the true positives. Returns (t, sensitivity, specificity)."""
    p_pos = np.asarray(p_pos, dtype=float)
    is_pos = np.asarray(is_pos, dtype=bool)
    n_pos = int(is_pos.sum())
    if n_pos == 0:
        raise ValueError("No positive examples in the calibration split.")
    needed = max(1, math.ceil(target_sensitivity * n_pos))
    t = float(np.sort(p_pos[is_pos])[::-1][needed - 1])
    flagged = p_pos >= t
    sens = float((flagged & is_pos).sum() / n_pos)
    n_neg = int((~is_pos).sum())
    spec = float((~flagged & ~is_pos).sum() / n_neg) if n_neg else float("nan")
    return t, sens, spec


def sensitivity_specificity(p_pos, is_pos, threshold):
    p_pos = np.asarray(p_pos, dtype=float)
    is_pos = np.asarray(is_pos, dtype=bool)
    flagged = p_pos >= threshold
    n_pos, n_neg = int(is_pos.sum()), int((~is_pos).sum())
    return {
        "threshold": float(threshold),
        "sensitivity": float((flagged & is_pos).sum() / n_pos) if n_pos else float("nan"),
        "specificity": float((~flagged & ~is_pos).sum() / n_neg) if n_neg else float("nan"),
        "n_positive": n_pos,
        "n_negative": n_neg,
    }


def classification_report(labels, probs, class_names):
    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support, roc_auc_score)

    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = probs.argmax(axis=1)
    k = len(class_names)
    prec, rec, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(k)), zero_division=0)

    aucs = {}
    for i, name in enumerate(class_names):
        y = (labels == i).astype(int)
        if 0 < y.sum() < len(y):
            aucs[name] = float(roc_auc_score(y, probs[:, i]))

    return {
        "n_samples": int(len(labels)),
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(np.mean(f1)),
        "macro_auc_ovr": float(np.mean(list(aucs.values()))) if aucs else float("nan"),
        "ece": expected_calibration_error(probs, labels),
        "nll": nll(probs, labels),
        "per_class": {
            name: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                   "support": int(support[i]), "auc": aucs.get(name)}
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(k))).tolist(),
    }


def collect_logits(model, loader, device, log_every=0):
    """Run `model` over a DataLoader -> (logits [N,C] float64, labels [N]).

    log_every > 0 logs progress every that many batches, so a stalled evaluation is visible."""
    import logging
    log = logging.getLogger("skinsense.eval")
    model.eval()
    device_type = torch.device(device).type
    all_logits, all_labels = [], []
    total = len(loader)
    # Mixed precision on CUDA (same as training): ~3-4x faster and far less GPU memory.
    with torch.no_grad(), torch.autocast(device_type=device_type, enabled=device_type == "cuda"):
        for i, (images, labels) in enumerate(loader, 1):
            out = model(images.to(device))
            all_logits.append(out.float().double().cpu().numpy())
            all_labels.append(labels.numpy())
            if log_every and (i % log_every == 0 or i == total):
                log.info("evaluation batch [%d/%d]", i, total)
    return np.concatenate(all_logits), np.concatenate(all_labels)
