import json

import numpy as np
import pytest

import app as app_module
from backend import metrics as M
from backend.fusion import MetadataFusion, build_features
from conftest import NEVI, FakeStore, image_bytes

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import train_fusion as tf  # noqa: E402

SPEC = {"age_mean": 50.0, "age_std": 10.0, "sites": tf.SITES}


def test_features_encode_age_sex_site_and_missing_values():
    lp = np.log(np.full((2, 7), 1 / 7))
    x = build_features(lp, [60.0, None], ["male", None], ["back", None], SPEC)
    assert x.shape == (2, 7 + 2 + 2 + len(tf.SITES))
    assert x[0, 7] == pytest.approx(1.0) and x[0, 8] == 0.0                   # age z-score, not missing
    assert x[1, 7] == 0.0 and x[1, 8] == 1.0                                  # missing age -> flag
    assert x[0, 9:11].tolist() == [1.0, 0.0] and x[1, 9:11].tolist() == [0.0, 0.0]
    assert x[0, 11 + tf.SITES.index("back")] == 1.0 and x[1, 11:].sum() == 0.0


def test_fusion_returns_a_probability_distribution_and_loads_optionally(tmp_path):
    k, f = 7, 7 + 2 + 2 + len(tf.SITES)
    spec = {**SPEC, "coef": np.random.default_rng(0).normal(0, 0.1, (k, f)).tolist(),
            "intercept": [0.0] * k, "melanoma_threshold": 0.2}
    path = tmp_path / "fusion.json"
    assert MetadataFusion.load(path) is None
    path.write_text(json.dumps(spec))
    fz = MetadataFusion.load(path)
    p = fz.predict(np.full(7, 1 / 7), age=70, sex="female", site="face")
    assert p.shape == (7,) and p.sum() == pytest.approx(1.0) and (p >= 0).all()
    assert MetadataFusion.has_context(None, None, None) is False and MetadataFusion.has_context(40, None, None)


def test_the_fusion_recipe_learns_a_site_effect_that_the_image_model_cannot_see():
    """Synthetic world: the class mix depends on body site, the 'image model' is only weakly informative.
    The fusion features must beat re-calibration alone on held-out data."""
    rng = np.random.default_rng(0)
    n, k = 3000, 7
    site_idx = rng.integers(0, len(tf.SITES), n)
    prefer = rng.integers(0, k, len(tf.SITES))                      # each site favours one class
    y = np.where(rng.random(n) < 0.6, prefer[site_idx], rng.integers(0, k, n))
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), y] += 0.8                                  # weak image signal
    lp = np.log(M.softmax(logits))
    sites = [tf.SITES[i] for i in site_idx]
    recal = build_features(lp, np.full(n, np.nan), [None] * n, [None] * n, SPEC)
    fused = build_features(lp, np.full(n, np.nan), [None] * n, sites, SPEC)
    tr, te = np.arange(n) < 2000, np.arange(n) >= 2000
    acc = lambda X: (tf.fit_logreg(X[tr], y[tr], 0.3).predict(X[te]) == y[te]).mean()
    assert acc(fused) > acc(recal) + 0.08


def test_api_uses_fusion_only_when_context_is_given(client, make_user, monkeypatch):
    k, f = 7, 7 + 2 + 2 + len(tf.SITES)
    coef = np.zeros((k, f))
    coef[:, :7] = np.eye(k)                                          # pass the image model through ...
    coef[4, 11 + tf.SITES.index("face")] = 6.0                       # ... but a face lesion strongly favours melanoma
    fz = MetadataFusion({**SPEC, "coef": coef.tolist(), "intercept": [0.0] * k, "melanoma_threshold": 0.5})
    monkeypatch.setattr(app_module, "fusion", fz)
    monkeypatch.setattr(app_module, "model_store", FakeStore(NEVI))
    _, h = make_user()
    post = lambda **extra: client.post("/predict", headers=h, files={"file": ("a.jpg", image_bytes(), "image/jpeg")},
                                       data=extra or None).json()
    plain = post()
    assert plain["context_used"] is False and plain["prediction"] == "Nevi"
    with_site = post(site="face")
    assert with_site["context_used"] is True
    assert with_site["melanoma_probability"] > plain["melanoma_probability"] + 20
    assert client.get("/health").json()["context_fusion"] is True
