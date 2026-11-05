import os
from pathlib import Path

import numpy as np
import pytest

import app as app_module
import database
from conftest import NEVI, NEVI_BUT_MEL_RISK, FakeStore, MissingStore, image_bytes
from test_care import make_clinician

PNG_MAGIC = bytes([0x89, 0x50, 0x4E, 0x47])


@pytest.fixture()
def fake_cam(monkeypatch):
    calls = {"n": 0, "target": None}

    def cam(model, tensor, target_idx, layers=None, use_amp=True):
        calls["n"] += 1
        calls["target"] = target_idx
        c = np.zeros((384, 384), dtype=np.float32)
        c[100:200, 100:200] = 1.0
        return c

    monkeypatch.setattr(app_module, "ensemble_cam", cam)
    return calls


def scan(client, headers):
    r = client.post("/predict", headers=headers, files={"file": ("a.jpg", image_bytes(), "image/jpeg")})
    assert r.status_code == 200
    return r.json()["scan_id"]


def uploads():
    return Path(os.environ["SKINSENSE_UPLOADS"])


def test_predict_returns_the_scan_id(client, make_user):
    _, h = make_user()
    sid = scan(client, h)
    assert client.get("/history", headers=h).json()[0]["id"] == sid


def test_owner_gets_a_png_and_it_is_cached_encrypted(client, make_user, fake_cam):
    _, h = make_user()
    sid = scan(client, h)
    r = client.get(f"/history/{sid}/heatmap", headers=h)
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content.startswith(PNG_MAGIC)
    assert fake_cam["n"] == 1
    again = client.get(f"/history/{sid}/heatmap", headers=h)
    assert again.content == r.content and fake_cam["n"] == 1, "second request must come from the cache"
    cached = [p for p in uploads().glob("*.cam.enc")]
    assert cached and not cached[-1].read_bytes().startswith(PNG_MAGIC), "cached heat-maps are encrypted at rest"


def test_heatmap_is_private(client, make_user, fake_cam):
    _, alice = make_user()
    _, bob = make_user()
    sid = scan(client, alice)
    assert client.get(f"/history/{sid}/heatmap").status_code == 401
    assert client.get(f"/history/{sid}/heatmap", headers=bob).status_code == 404


def test_the_explained_class_is_the_one_shown_to_the_user(client, make_user, fake_cam, monkeypatch):
    _, h = make_user()
    sid = scan(client, h)                                           # Nevi
    client.get(f"/history/{sid}/heatmap", headers=h)
    assert fake_cam["target"] == 5
    monkeypatch.setattr(app_module, "model_store", FakeStore(NEVI_BUT_MEL_RISK))
    sid2 = scan(client, h)                                          # escalated -> "Melanoma (High-Priority ...)"
    client.get(f"/history/{sid2}/heatmap", headers=h)
    assert fake_cam["target"] == 4


def test_deleting_a_scan_removes_its_heatmap_file(client, make_user, fake_cam):
    _, h = make_user()
    sid = scan(client, h)
    client.get(f"/history/{sid}/heatmap", headers=h)
    assert list(uploads().glob("*.cam.enc"))
    before = len(list(uploads().glob("*.cam.enc")))
    assert client.delete(f"/history/{sid}", headers=h).status_code == 200
    assert len(list(uploads().glob("*.cam.enc"))) == before - 1


def test_heatmap_needs_a_loaded_model(client, make_user, monkeypatch):
    _, h = make_user()
    sid = scan(client, h)
    monkeypatch.setattr(app_module, "model_store", MissingStore())
    assert client.get(f"/history/{sid}/heatmap", headers=h).status_code == 503


def test_clinician_heatmap_requires_consent_and_is_audited(client, make_user, fake_cam):
    clin_name, clin = make_clinician(client, make_user)
    _, pat = make_user()
    sid = scan(client, pat)
    with database.SessionLocal() as db:
        patient_id = db.query(database.History).filter_by(id=sid).first().user_id
    url = f"/clinician/patients/{patient_id}/history/{sid}/heatmap"
    assert client.get(url, headers=clin).status_code == 403                       # no consent yet
    assert client.get(url, headers=pat).status_code == 403                        # patients cannot use it
    client.post("/sharing", headers=pat, json={"clinician_username": clin_name})
    r = client.get(url, headers=clin)
    assert r.status_code == 200 and r.content.startswith(PNG_MAGIC)
    assert "viewed_heatmap" in [a["action"] for a in client.get("/audit", headers=pat).json()]
    client.delete(f"/sharing/{client.get('/sharing', headers=pat).json()[0]['id']}", headers=pat)
    assert client.get(url, headers=clin).status_code == 403                       # revoked
