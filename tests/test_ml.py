import json

import numpy as np
import pytest

from backend import data_utils, metrics as M
from inference import decide
import settings

CLASSES = settings.CLASSES
CFG = {"melanoma_threshold": 0.04, "min_confidence": 0.35, "temperature": 1.0}


def onehot_probs(top, mel=0.0, top_p=0.9):
    p = np.full(7, (1 - top_p - mel) / 5)
    p[top] = top_p
    p[4] = mel if top != 4 else top_p
    return p / p.sum()


# ---------------------------------------------------------------- decision logic
def test_decide_normal_prediction():
    d = decide(onehot_probs(5, mel=0.01), CLASSES, CFG)
    assert d["prediction"] == "Nevi" and not d["escalated"] and d["class_name"] == "Nevi"


def test_decide_escalates_when_melanoma_prob_over_threshold():
    d = decide(onehot_probs(5, mel=0.10, top_p=0.80), CLASSES, CFG)
    assert d["escalated"] and d["class_name"] == "Melanoma"
    assert d["confidence"] == pytest.approx(10.0, abs=0.5) and d["top_prediction"] == "Nevi"


def test_decide_does_not_escalate_when_melanoma_is_already_top():
    d = decide(onehot_probs(4), CLASSES, CFG)
    assert d["prediction"] == "Melanoma" and not d["escalated"]


def test_decide_uncertain():
    d = decide(np.full(7, 1 / 7), CLASSES, {**CFG, "melanoma_threshold": 0.9})
    assert d["uncertain"] and d["class_name"] is None


def test_threshold_is_configurable_not_hardcoded():
    probs = onehot_probs(5, mel=0.10, top_p=0.80)
    assert decide(probs, CLASSES, {**CFG, "melanoma_threshold": 0.5})["escalated"] is False


def test_calibration_file_is_loaded(tmp_path, monkeypatch):
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"temperature": 2.5, "melanoma_threshold": 0.2, "source": "unit-test"}))
    monkeypatch.setattr(settings, "CALIBRATION_PATH", f)
    cfg = settings.load_calibration()
    assert cfg["temperature"] == 2.5 and cfg["melanoma_threshold"] == 0.2 and cfg["min_confidence"] == 0.35


# ---------------------------------------------------------------- metrics
def test_softmax_sums_to_one_and_temperature_flattens():
    z = np.array([[2.0, 0.0, -1.0]])
    assert M.softmax(z).sum() == pytest.approx(1.0)
    assert M.softmax(z, 5.0).max() < M.softmax(z, 1.0).max()


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    n, k = 4000, 7
    labels = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    right = rng.random(n) < 0.6                       # model is only 60 % right ...
    pred = np.where(right, labels, rng.integers(0, k, n))
    logits[np.arange(n), pred] += 8.0                 # ... but extremely confident
    T = M.fit_temperature(logits, labels)
    assert T > 1.5
    assert M.nll(M.softmax(logits, T), labels) < M.nll(M.softmax(logits), labels)
    assert M.expected_calibration_error(M.softmax(logits, T), labels) < \
        M.expected_calibration_error(M.softmax(logits), labels)


def test_ece_is_zero_for_perfect_confident_predictions():
    probs = np.eye(3)[[0, 1, 2, 0]]
    assert M.expected_calibration_error(probs, np.array([0, 1, 2, 0])) == pytest.approx(0.0)


def test_pick_sensitivity_threshold_hits_target():
    rng = np.random.default_rng(1)
    is_pos = rng.random(2000) < 0.1
    p = np.where(is_pos, rng.beta(5, 2, 2000), rng.beta(1, 8, 2000))
    t, sens, spec = M.pick_sensitivity_threshold(p, is_pos, 0.90)
    assert sens >= 0.90 and 0.5 < spec < 1.0
    again = M.sensitivity_specificity(p, is_pos, t)
    assert again["sensitivity"] == pytest.approx(sens)


def test_classification_report_shapes_and_balanced_accuracy():
    labels = np.array([0] * 90 + [1] * 10)
    probs = np.tile([0.9, 0.1], (100, 1))             # always predicts class 0
    rep = M.classification_report(labels, probs, ["a", "b"])
    assert rep["accuracy"] == pytest.approx(0.9)
    assert rep["balanced_accuracy"] == pytest.approx(0.5)    # accuracy hides the failure on class b
    assert rep["per_class"]["b"]["recall"] == 0.0
    assert np.array(rep["confusion_matrix"]).sum() == 100


# ---------------------------------------------------------------- splits (leakage fix)
def test_grouped_split_has_no_lesion_overlap_and_is_stratified():
    rng = np.random.default_rng(3)
    labels, groups = [], []
    for lesion in range(600):                         # each lesion has 1-4 photos, all same class
        cls = int(rng.choice(7, p=[.03, .05, .11, .01, .11, .67, .02]))
        for _ in range(int(rng.integers(1, 5))):
            labels.append(cls)
            groups.append(f"L{lesion}")
    splits = data_utils.make_splits(labels, groups, seed=1)
    data_utils.check_no_overlap(splits, groups)       # raises on leakage
    n = len(labels)
    assert sum(len(v) for v in splits.values()) == n
    assert 0.6 < len(splits["train"]) / n < 0.8
    assert 0.1 < len(splits["val"]) / n < 0.2 and 0.1 < len(splits["test"]) / n < 0.2
    assert len({*splits["train"], *splits["val"], *splits["test"]}) == n


def test_check_no_overlap_detects_leakage():
    groups = ["a", "a", "b"]
    with pytest.raises(AssertionError):
        data_utils.check_no_overlap({"train": [0], "val": [1], "test": [2]}, groups)


def test_per_image_split_would_leak_demonstration():
    """Why the fix matters: a naive random split puts the same lesion on both sides."""
    groups = np.repeat(np.arange(300), 3)
    perm = np.random.default_rng(0).permutation(len(groups))
    train, test = perm[:700], perm[700:]
    assert set(groups[train]) & set(groups[test])
