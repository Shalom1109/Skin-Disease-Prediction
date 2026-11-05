"""Grad-CAM heat-maps for the ensemble: which parts of the photo pushed the model towards a class.

Each member network is explained at its last feature stage (ConvNeXt and MaxViT: a spatial feature map; ViT: the
patch tokens reshaped to a grid), the three maps are weighted like the ensemble itself, and the result is overlaid on
the exact 384x384 view the model saw. A heat-map is an explanation aid, NOT a diagnosis: it can highlight a hair, a
ruler or a border instead of the lesion, which is precisely why clinicians should look at it.
"""
import io
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from inference import IMG_SIZE

_VIEW = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE)])


def default_layers(model):
    """(name -> (module whose output is explained, ensemble weight, kind)) for SkinSenseEnsemble."""
    return {
        "convnext": (model.convnext.stages[-1], model.w1, "conv"),
        "vit": (model.vit.blocks[-1], model.w2, "tokens"),
        "maxvit": (model.maxvit.stages[-1], model.w3, "conv"),
    }


def _to_grid(t, kind, n_prefix):
    """-> [1, C, h, w] from either a feature map or a token sequence."""
    if kind == "conv":
        return t
    tokens = t[:, n_prefix:]                                    # drop the class token(s)
    n = int(math.isqrt(tokens.shape[1]))
    return tokens.reshape(1, n, n, -1).permute(0, 3, 1, 2)


def ensemble_cam(model, tensor, target_idx, layers=None, use_amp=True):
    """Grad-CAM for class `target_idx`. `tensor` is the preprocessed [1,3,H,W] input. Returns float32 [H,W] in 0..1."""
    layers = layers or default_layers(model)
    device = next(model.parameters()).device
    x = tensor.to(device).float().requires_grad_(True)          # makes the whole graph differentiable
    captured, handles = {}, []
    for name, (module, _w, _k) in layers.items():
        def hook(_m, _inp, out, name=name):
            captured[name] = out
            out.retain_grad()
        handles.append(module.register_forward_hook(hook))
    n_prefix = getattr(getattr(model, "vit", None), "num_prefix_tokens", 1)
    model.eval()
    try:
        with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(x)
            score = logits[0, target_idx].float()
        model.zero_grad(set_to_none=True)
        score.backward()
        total = torch.zeros(tensor.shape[-2:], device=device)
        for name, (_m, weight, kind) in layers.items():
            act = _to_grid(captured[name].detach().float(), kind, n_prefix)
            grad = _to_grid(captured[name].grad.float(), kind, n_prefix)
            cam = F.relu((grad.mean(dim=(2, 3), keepdim=True) * act).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=tuple(tensor.shape[-2:]), mode="bilinear", align_corners=False)[0, 0]
            total += weight * cam / (cam.max() + 1e-8)
    finally:
        for h in handles:
            h.remove()
    total = total / (total.max() + 1e-8)
    return total.cpu().numpy().astype(np.float32)


def _jet(v):
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_png(image: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> bytes:
    """Blend the heat-map over the 384x384 view of `image` that the model actually saw."""
    view = _VIEW(image.convert("RGB")).resize(cam.shape[::-1])
    heat = Image.fromarray(_jet(cam))
    # transparent where the model paid little attention, so the lesion stays visible
    mask = Image.fromarray((np.clip(cam * 1.4, 0, 1) * 255 * alpha).astype(np.uint8))
    out = Image.composite(heat, view, mask)
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()
