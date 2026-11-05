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


# ---------------------------------------------------------------- uncertainty, CIs, per-model outputs
def test_disagreement_is_zero_when_networks_agree_and_large_when_they_do_not():
    same = np.tile(np.array([[5.0, 0, 0, 0, 0, 0, 0]]), (4, 3, 1))
    assert M.disagreement(same) == pytest.approx(np.zeros(4))
    split = np.zeros((1, 3, 7))
    split[0, 0, 0] = split[0, 1, 1] = split[0, 2, 2] = 20.0
    assert M.disagreement(split)[0] > 0.9


def test_error_detection_auroc_perfect_and_random():
    probs = np.tile([0.9, 0.1], (100, 1))
    labels = np.array([0] * 50 + [1] * 50)           # second half is wrong
    perfect = np.array([0.0] * 50 + [1.0] * 50)
    assert M.error_detection_auroc(labels, probs, perfect) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    assert 0.35 < M.error_detection_auroc(labels, probs, rng.random(100)) < 0.65


def test_risk_coverage_improves_accuracy_when_abstaining_on_the_uncertain():
    labels = np.array([0] * 80 + [1] * 20)
    probs = np.tile([0.9, 0.1], (100, 1))            # always predicts class 0 -> class-1 cases are wrong
    uncertainty = np.array([0.1] * 80 + [0.9] * 20)  # the wrong ones are flagged uncertain
    rows = {r["coverage"]: r for r in M.risk_coverage(labels, probs, uncertainty, mel_idx=1)}
    assert rows[1.0]["accuracy"] == pytest.approx(0.8)
    assert rows[0.8]["accuracy"] == pytest.approx(1.0)
    assert rows[0.8]["melanomas_kept"] == pytest.approx(0.0)       # abstaining drops those cases


def test_bootstrap_ci_brackets_the_point_estimate_and_keeps_lesions_together():
    rng = np.random.default_rng(1)
    n, k = 600, 7
    labels = rng.integers(0, k, n)
    probs = np.full((n, k), 0.02)
    right = rng.random(n) < 0.7
    pred = np.where(right, labels, (labels + 1) % k)
    probs[np.arange(n), pred] = 0.88
    probs /= probs.sum(axis=1, keepdims=True)
    groups = np.repeat(np.arange(n // 2), 2)
    ci = M.bootstrap_ci(labels, probs, groups, mel_idx=4, mel_threshold=0.1, n_boot=200, seed=0)
    acc = (probs.argmax(1) == labels).mean()
    assert ci["accuracy"]["low"] < acc < ci["accuracy"]["high"]
    assert 0.0 < ci["accuracy"]["high"] - ci["accuracy"]["low"] < 0.15
    assert set(ci) == {"accuracy", "balanced_accuracy", "macro_f1", "melanoma_auc",
                       "melanoma_sensitivity", "melanoma_specificity"}


def test_collect_outputs_tta_averages_views_and_matches_forward():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    class Tiny(torch.nn.Module):
        w1, w2, w3 = 0.4, 0.3, 0.3

        def __init__(self):
            super().__init__()
            self.lin = torch.nn.ModuleList([torch.nn.Linear(12, 7) for _ in range(3)])

        def forward_parts(self, x):
            flat = x.flatten(1)
            return tuple(l(flat) for l in self.lin)

        def forward(self, x):
            a, b, c = self.forward_parts(x)
            return self.w1 * a + self.w2 * b + self.w3 * c

    torch.manual_seed(0)
    model = Tiny()
    data = TensorDataset(torch.randn(6, 3, 2, 2), torch.arange(6) % 7)
    loader = DataLoader(data, batch_size=3)
    ens, parts, labels = M.collect_outputs(model, loader, "cpu", tta=False)
    assert ens.shape == (6, 7) and parts.shape == (6, 3, 7) and labels.tolist() == list(range(6))
    x = data.tensors[0]
    assert np.allclose(ens, model(x).detach().numpy(), atol=1e-5)
    ens_tta, _, _ = M.collect_outputs(model, loader, "cpu", tta=True)
    expected = np.mean([model(v).detach().numpy() for v in (x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2))], axis=0)
    assert np.allclose(ens_tta, expected, atol=1e-5)


# ---------------------------------------------------------------- Grad-CAM heat-maps
def _tiny_ensemble():
    """A miniature model with the same hook points as the real ensemble (positive weights, no biases)."""
    import torch
    from torch import nn

    class Branch(nn.Module):
        def __init__(self):
            super().__init__()
            self.stages = nn.ModuleList([nn.Conv2d(3, 4, 1, bias=False)])
            self.fc = nn.Linear(4, 7, bias=False)
            with torch.no_grad():
                self.stages[-1].weight.abs_()
                self.fc.weight.abs_()

        def forward(self, x):
            return self.fc(self.stages[-1](x).mean(dim=(2, 3)))

    class Tiny(nn.Module):
        w1, w2, w3 = 1.0, 0.0, 0.0

        def __init__(self):
            super().__init__()
            self.convnext, self.vit, self.maxvit = Branch(), Branch(), Branch()

        def forward(self, x):
            return self.w1 * self.convnext(x) + self.w2 * self.vit(x) + self.w3 * self.maxvit(x)

    torch.manual_seed(0)
    return Tiny()


def test_gradcam_follows_the_image_content_that_drives_the_score():
    import torch
    import explain

    model = _tiny_ensemble()
    layers = {"convnext": (model.convnext.stages[-1], 1.0, "conv")}
    x = torch.zeros(1, 3, 32, 32)
    x[:, :, :16, :16] = 1.0                       # the only "lesion" is in the top-left quadrant
    cam = explain.ensemble_cam(model, x, target_idx=2, layers=layers)
    assert cam.shape == (32, 32) and cam.dtype == np.float32
    assert 0.0 <= cam.min() and cam.max() == pytest.approx(1.0)
    top_left = cam[:16, :16].mean()
    rest = np.concatenate([cam[:16, 16:].ravel(), cam[16:].ravel()]).mean()
    assert top_left > 5 * rest + 1e-6, (top_left, rest)


def test_gradcam_token_layers_are_reshaped_to_a_grid():
    import torch
    import explain
    t = torch.randn(1, 1 + 16, 8)                  # class token + a 4x4 patch grid
    grid = explain._to_grid(t, "tokens", n_prefix=1)
    assert grid.shape == (1, 8, 4, 4)


def test_overlay_is_a_valid_png_of_the_model_view_size():
    import io
    import explain
    from PIL import Image
    cam = np.zeros((384, 384), dtype=np.float32)
    cam[100:200, 100:200] = 1.0
    png = explain.overlay_png(Image.new("RGB", (640, 480), (190, 140, 120)), cam)
    img = Image.open(io.BytesIO(png))
    assert img.format == "PNG" and img.size == (384, 384)
