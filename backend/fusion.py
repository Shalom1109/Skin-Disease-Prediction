"""Late fusion of the image model's probabilities with optional patient context (age, sex, body site).

A small multinomial logistic regression, trained on the VALIDATION split only (the image networks never saw it),
takes  [log calibrated probabilities (7), age, age-missing flag, sex, body-site one-hot]  and returns new class
probabilities. It is used only when the user supplied at least one context field AND backend/fusion.json exists.
The file is produced by scripts/train_fusion.py, which also reports whether fusion actually helped - ship it only
if it did.
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np

SEXES = ["male", "female"]


def build_features(log_probs, age, sex, site, spec):
    """-> [N, F] feature matrix. `age` may contain NaN, `sex`/`site` may be None or unknown strings."""
    log_probs = np.atleast_2d(np.asarray(log_probs, dtype=float))
    n = len(log_probs)
    age = np.asarray([np.nan if a is None else a for a in (age if np.ndim(age) else [age] * n)], dtype=float)
    missing = np.isnan(age)
    age_z = np.where(missing, 0.0, (np.nan_to_num(age) - spec["age_mean"]) / spec["age_std"])
    sex = list(sex) if np.ndim(sex) else [sex] * n
    site = list(site) if np.ndim(site) else [site] * n
    sex_oh = np.array([[1.0 if (s or "").lower() == v else 0.0 for v in SEXES] for s in sex])
    site_oh = np.array([[1.0 if (s or "").lower() == v else 0.0 for v in spec["sites"]] for s in site])
    return np.hstack([log_probs, age_z[:, None], missing[:, None].astype(float), sex_oh, site_oh])


class MetadataFusion:
    def __init__(self, spec: dict):
        self.spec = spec
        self.coef = np.asarray(spec["coef"], dtype=float)           # [K, F]
        self.intercept = np.asarray(spec["intercept"], dtype=float)  # [K]
        self.melanoma_threshold = float(spec["melanoma_threshold"])

    @classmethod
    def load(cls, path) -> Optional["MetadataFusion"]:
        path = Path(path)
        if not path.exists():
            return None
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def predict(self, base_probs, age=None, sex=None, site=None) -> np.ndarray:
        """base_probs: calibrated probabilities from the image model [K]. Returns fused probabilities [K]."""
        x = build_features(np.log(np.clip(base_probs, 1e-9, 1.0)), age, sex, site, self.spec)
        z = x @ self.coef.T + self.intercept
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return (e / e.sum(axis=1, keepdims=True))[0]

    @staticmethod
    def has_context(age, sex, site) -> bool:
        return age is not None or bool(sex) or bool(site)
