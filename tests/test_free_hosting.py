import pytest
import torch

import app as app_module
from conftest import image_bytes
from test_heatmap import fake_cam, scan  # noqa: F401  (fixture + helper)


# ---------------------------------------------------------------- weights from a Hugging Face repo
def test_weights_are_fetched_from_the_hub_when_configured(monkeypatch):
    import huggingface_hub
    seen = {}

    def fake_download(repo_id, filename, token=None, **kwargs):
        seen.update(repo=repo_id, file=filename, token=token)
        return "/cache/model.pth"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.delenv("SKINSENSE_WEIGHTS_REPO", raising=False)
    assert app_module.fetch_weights_from_hub() is None                      # not configured -> does nothing
    monkeypatch.setenv("SKINSENSE_WEIGHTS_REPO", "someone/skinsense-weights")
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    assert app_module.fetch_weights_from_hub() == "/cache/model.pth"
    assert seen == {"repo": "someone/skinsense-weights", "file": "skin_sense_ensemble_best.pth", "token": "hf_secret"}


def test_model_store_uses_downloaded_weights_when_none_are_on_disk(monkeypatch):
    import model as model_module
    seen = {}

    def fake_get_model(num_classes, checkpoint_path=None):
        seen["checkpoint"] = checkpoint_path
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(model_module, "get_model", fake_get_model)
    monkeypatch.setattr(app_module, "find_checkpoint", lambda: None)
    monkeypatch.setattr(app_module, "fetch_weights_from_hub", lambda: "/cache/model.pth")
    app_module.ModelStore().get()
    assert seen["checkpoint"] == "/cache/model.pth"


def test_no_weights_anywhere_is_still_a_clear_503_not_random_output(monkeypatch):
    monkeypatch.setattr(app_module, "find_checkpoint", lambda: None)
    monkeypatch.setattr(app_module, "fetch_weights_from_hub", lambda: None)
    with pytest.raises(app_module.ModelUnavailable):
        app_module.ModelStore().get()


# ---------------------------------------------------------------- heat-maps on CPU-only hosts
def test_heatmaps_follow_the_gpu_by_default_and_can_be_forced(monkeypatch):
    monkeypatch.setenv("SKINSENSE_HEATMAP", "auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert app_module.heatmaps_enabled() is False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert app_module.heatmaps_enabled() is True
    monkeypatch.setenv("SKINSENSE_DEVICE", "cpu")             # forced CPU inference -> no heat-maps either
    assert app_module.heatmaps_enabled() is False
    monkeypatch.delenv("SKINSENSE_DEVICE")
    monkeypatch.setenv("SKINSENSE_HEATMAP", "on")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert app_module.heatmaps_enabled() is True
    monkeypatch.setenv("SKINSENSE_HEATMAP", "off")
    assert app_module.heatmaps_enabled() is False


def test_a_cpu_host_turns_heatmaps_off_everywhere_and_says_so(client, make_user, fake_cam, monkeypatch):
    _, h = make_user()
    sid = scan(client, h)
    assert client.get("/health").json()["heatmaps"] is True
    monkeypatch.setenv("SKINSENSE_HEATMAP", "off")
    assert client.get("/health").json()["heatmaps"] is False
    r = client.get(f"/history/{sid}/heatmap", headers=h)
    assert r.status_code == 503 and "turned off" in r.json()["detail"]
    assert fake_cam["n"] == 0, "no expensive computation may start when heat-maps are off"


# ---------------------------------------------------------------- being shown inside another site's page
def test_page_can_only_be_embedded_by_explicitly_allowed_sites(client, monkeypatch):
    monkeypatch.delenv("SKINSENSE_FRAME_ANCESTORS", raising=False)
    h = client.get("/health").headers
    assert h["x-frame-options"] == "DENY" and "content-security-policy" not in h

    monkeypatch.setenv("SKINSENSE_FRAME_ANCESTORS", "https://huggingface.co")
    h = client.get("/health").headers
    assert h["content-security-policy"] == "frame-ancestors 'self' https://huggingface.co"
    assert "x-frame-options" not in h

    monkeypatch.setenv("SKINSENSE_FRAME_ANCESTORS", "https://ok.example; script-src *")    # injection attempt
    h = client.get("/health").headers
    assert h["x-frame-options"] == "DENY" and "content-security-policy" not in h
