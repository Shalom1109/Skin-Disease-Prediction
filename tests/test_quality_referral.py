import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import quality
from conftest import NEVI_BUT_MEL_RISK, UNSURE, FakeStore
from inference import decide, referral
import app as app_module
import settings

CLASSES = settings.CLASSES
CFG = {"melanoma_threshold": 0.04, "min_confidence": 0.35, "temperature": 1.0}


def textured(size=(640, 480), seed=0):
    """A skin-toned, textured image that looks 'sharp' to the gate."""
    rng = np.random.default_rng(seed)
    base = np.array([190, 140, 120], dtype=float)
    noise = rng.normal(0, 28, (size[1], size[0], 3))
    return Image.fromarray(np.clip(base + noise, 0, 255).astype("uint8"))


# ---------------------------------------------------------------- quality gate
def test_a_sharp_well_lit_photo_passes():
    res = quality.assess_quality(textured())
    assert res["ok"] and res["issues"] == [] and res["tips"] == []


def test_blur_darkness_brightness_and_size_are_caught():
    img = textured()
    assert "blurry" in quality.assess_quality(img.filter(ImageFilter.GaussianBlur(8)))["issues"]
    assert "too_dark" in quality.assess_quality(ImageEnhance.Brightness(img).enhance(0.2))["issues"]
    assert "too_bright" in quality.assess_quality(ImageEnhance.Brightness(img).enhance(2.8))["issues"]
    assert "low_resolution" in quality.assess_quality(img.resize((120, 90)))["issues"]


def test_every_issue_has_a_tip_and_metrics_are_reported():
    res = quality.assess_quality(textured().filter(ImageFilter.GaussianBlur(8)))
    assert len(res["tips"]) == len(res["issues"]) > 0
    assert {"sharpness", "brightness", "contrast", "width", "height"} <= set(res["metrics"])


# ---------------------------------------------------------------- referral
def probs_for(top, mel=0.01, top_p=0.9):
    p = np.full(7, (1 - top_p - mel) / 5)
    p[top] = top_p
    if top != 4:
        p[4] = mel
    return p / p.sum()


def test_referral_levels_follow_the_clinical_risk_of_the_class():
    names = {n: i for i, n in enumerate(CLASSES)}
    assert referral(decide(probs_for(names["Melanoma"]), CLASSES, CFG))["level"] == "priority"
    assert referral(decide(probs_for(names["Basal Cell Carcinoma"]), CLASSES, CFG))["level"] == "priority"
    assert referral(decide(probs_for(names["Actinic Keratoses"]), CLASSES, CFG))["level"] == "soon"
    for benign in ("Nevi", "Benign Keratosis", "Dermatofibroma", "Vascular"):
        assert referral(decide(probs_for(names[benign]), CLASSES, CFG))["level"] == "routine", benign
    assert referral(decide(np.full(7, 1 / 7), CLASSES, {**CFG, "melanoma_threshold": 0.9}))["level"] == "inconclusive"


def test_an_escalation_is_always_a_priority_referral_even_if_nevus_is_on_top():
    d = decide(probs_for(5, mel=0.12, top_p=0.8), CLASSES, CFG)
    assert d["escalated"] and referral(d)["level"] == "priority"


def test_referral_always_carries_the_safety_advice_and_notes_poor_quality():
    d = decide(probs_for(5), CLASSES, CFG)
    r = referral(d, {"ok": False})
    assert "see a doctor whatever this result says" in r["advice"]
    assert "retaking" in r["message"]
    assert "retaking" not in referral(decide(probs_for(4), CLASSES, CFG), {"ok": False})["message"]   # priority unchanged


# ---------------------------------------------------------------- API
def jpeg(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


def test_predict_returns_referral_and_quality(client, make_user):
    _, h = make_user()
    d = client.post("/predict", headers=h, files={"file": ("a.jpg", jpeg(textured()), "image/jpeg")}).json()
    assert d["referral"]["level"] == "routine" and d["quality"]["ok"] is True
    assert "advice" in d["referral"]


def test_predict_flags_a_blurry_photo_but_still_answers(client, make_user):
    _, h = make_user()
    blurry = textured().filter(ImageFilter.GaussianBlur(8))
    d = client.post("/predict", headers=h, files={"file": ("b.jpg", jpeg(blurry), "image/jpeg")}).json()
    assert "blurry" in d["quality"]["issues"] and d["prediction"]


def test_melanoma_risk_gives_priority_referral_through_the_api(client, make_user, monkeypatch):
    monkeypatch.setattr(app_module, "model_store", FakeStore(NEVI_BUT_MEL_RISK))
    _, h = make_user()
    d = client.post("/predict", headers=h, files={"file": ("c.jpg", jpeg(textured()), "image/jpeg")}).json()
    assert d["referral"]["level"] == "priority"


def test_quality_endpoint_needs_login_and_gives_instant_feedback(client, make_user):
    assert client.post("/quality", files={"file": ("a.jpg", jpeg(textured()), "image/jpeg")}).status_code == 401
    _, h = make_user()
    r = client.post("/quality", headers=h, files={"file": ("a.jpg", jpeg(textured().resize((100, 80))), "image/jpeg")})
    assert r.status_code == 200 and "low_resolution" in r.json()["issues"]
