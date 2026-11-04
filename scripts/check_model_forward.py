"""Builds the ensemble (no downloads) and runs one dummy 384x384 forward pass.

    python scripts/check_model_forward.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.model import SkinSenseEnsemble  # noqa: E402

model = SkinSenseEnsemble(num_classes=7, pretrained=False).eval()
with torch.no_grad():
    out = model(torch.randn(1, 3, 384, 384))
assert out.shape == (1, 7), out.shape
print("OK - output shape:", tuple(out.shape))
