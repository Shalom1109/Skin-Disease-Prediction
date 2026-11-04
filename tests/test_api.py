import os
from pathlib import Path

from PIL import Image

import app as app_module
from conftest import NEVI_BUT_MEL_RISK, UNSURE, FakeStore, MissingStore, image_bytes


def upload(client, headers, data=None, name="lesion.jpg", ctype="image/jpeg"):
    data = image_bytes() if data is None else data
    return client.post("/predict", headers=headers, files={"file": (name, data, ctype)})


# ---------------------------------------------------------------- auth
def test_register_rules(client):
    def reg(u, p):
        return client.post("/register", data={"username": u, "password": p})

    assert reg("ab", "Passw0rd!x").status_code == 400            # too short
    assert reg("bad name!", "Passw0rd!x").status_code == 400     # illegal chars
    assert reg("okname1", "Sh0rt!").status_code == 400           # < 8 chars
    assert reg("okname1", "NOLOWER123!").status_code == 400
    assert reg("okname1", "NoDigits!!").status_code == 400
    assert reg("okname1", "NoSpecial123").status_code == 400
    assert reg("okname1", "Passw0rd!x").status_code == 200
    assert reg("OKNAME1", "Passw0rd!x").status_code == 400       # case-insensitive duplicate


def test_login_error_does_not_reveal_whether_user_exists(client, make_user):
    name, _ = make_user()
    wrong_pw = client.post("/login", data={"username": name, "password": "WrongPass1!"})
    no_user = client.post("/login", data={"username": "nobody_here", "password": "WrongPass1!"})
    assert wrong_pw.status_code == no_user.status_code == 401
    assert wrong_pw.json() == no_user.json()


def test_login_is_rate_limited(client, make_user):
    name, _ = make_user()
    codes = [client.post("/login", data={"username": name, "password": "bad"}).status_code for _ in range(6)]
    assert codes[:5] == [401] * 5 and codes[5] == 429
    # even the right password is refused while locked out
    assert client.post("/login", data={"username": name, "password": "Passw0rd!x"}).status_code == 429


def test_protected_routes_need_a_token(client):
    for method, path in (("get", "/history"), ("get", "/profile"), ("post", "/chat"), ("post", "/predict")):
        assert getattr(client, method)(path).status_code == 401
    assert client.get("/history", headers={"Authorization": "Bearer garbage"}).status_code == 401


# ---------------------------------------------------------------- predict
def test_predict_happy_path(client, make_user):
    _, h = make_user()
    r = upload(client, h)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["prediction"] == "Nevi" and not d["escalated"] and not d["uncertain"]
    assert 0 < d["confidence"] <= 100
    assert set(d["probabilities"]) == set(app_module.settings.CLASSES)
    assert abs(sum(d["probabilities"].values()) - 100) < 1.0
    for key in ("description", "symptoms", "recommendations", "dos", "donts", "disclaimer"):
        assert d[key]
    assert len(client.get("/history", headers=h).json()) == 1


def test_melanoma_escalation_is_explained_and_uses_real_probability(client, make_user, monkeypatch):
    monkeypatch.setattr(app_module, "model_store", FakeStore(NEVI_BUT_MEL_RISK))
    _, h = make_user()
    d = upload(client, h).json()
    assert d["escalated"] and d["prediction"] == app_module.settings.ESCALATED_LABEL
    assert d["confidence"] == d["melanoma_probability"]          # not the nevus confidence
    assert d["top_prediction"] == "Nevi" and "screening flag" in d["note"]
    assert "Melanoma" in d["description"]                          # KB entry found (old code missed it)


def test_low_confidence_is_reported_as_uncertain(client, make_user, monkeypatch):
    monkeypatch.setattr(app_module, "model_store", FakeStore(UNSURE))
    monkeypatch.setitem(app_module.calibration, "melanoma_threshold", 0.99)
    _, h = make_user()
    d = upload(client, h).json()
    assert d["uncertain"] and d["prediction"] == app_module.settings.UNCERTAIN_LABEL


def test_predict_without_weights_returns_503_not_random_output(client, make_user, monkeypatch):
    monkeypatch.setattr(app_module, "model_store", MissingStore())
    _, h = make_user()
    r = upload(client, h)
    assert r.status_code == 503 and "weights" in r.json()["detail"].lower()


def test_upload_validation(client, make_user, monkeypatch):
    _, h = make_user()
    assert upload(client, h, data=b"not an image").status_code == 400
    assert upload(client, h, data=b"").status_code == 400
    assert upload(client, h, data=image_bytes("GIF"), name="a.gif").status_code == 415
    monkeypatch.setattr(app_module.settings, "MAX_UPLOAD_BYTES", 1000)
    assert upload(client, h, data=image_bytes(size=(400, 400))).status_code == 413


def test_stored_image_has_no_exif_and_is_a_jpeg(client, make_user):
    _, h = make_user()
    assert upload(client, h, data=image_bytes("JPEG", exif=True)).status_code == 200
    files = list(Path(os.environ["SKINSENSE_UPLOADS"]).glob("*.jpg"))
    assert files
    with Image.open(files[-1]) as img:
        assert img.format == "JPEG" and not img.getexif()


# ---------------------------------------------------------------- privacy
def test_images_are_private_to_their_owner(client, make_user):
    _, alice = make_user()
    _, bob = make_user()
    upload(client, alice)
    item = client.get("/history", headers=alice).json()[0]
    assert item["image_url"] == f"/history/{item['id']}/image"

    assert client.get(item["image_url"], headers=alice).status_code == 200
    assert client.get(item["image_url"], headers=bob).status_code == 404
    assert client.get(item["image_url"]).status_code == 401
    # the old public /uploads mount is gone
    stored = next(Path(os.environ["SKINSENSE_UPLOADS"]).glob("*.jpg")).name
    assert client.get(f"/uploads/{stored}").status_code == 404
    assert client.delete(f"/history/{item['id']}", headers=bob).status_code == 404


def test_delete_history_item_removes_row_and_file(client, make_user):
    _, h = make_user()
    uploads = Path(os.environ["SKINSENSE_UPLOADS"])
    before = set(uploads.glob("*.jpg"))
    upload(client, h)
    item = client.get("/history", headers=h).json()[0]
    (path,) = set(uploads.glob("*.jpg")) - before      # exactly the file this upload created
    assert path.exists()
    assert client.delete(f"/history/{item['id']}", headers=h).status_code == 200
    assert client.get("/history", headers=h).json() == []
    assert not path.exists()


def test_delete_account_erases_everything(client, make_user):
    name, h = make_user()
    upload(client, h)
    uploads = Path(os.environ["SKINSENSE_UPLOADS"])
    before = set(uploads.glob("*.jpg"))
    assert client.request("DELETE", "/account", headers=h, data={"password": "wrong"}).status_code == 401
    assert client.request("DELETE", "/account", headers=h, data={"password": "Passw0rd!x"}).status_code == 200
    assert len(set(uploads.glob("*.jpg"))) == len(before) - 1
    assert client.post("/login", data={"username": name, "password": "Passw0rd!x"}).status_code == 401


# ---------------------------------------------------------------- profile / chat / misc
def test_profile_update_and_clear(client, make_user):
    _, h = make_user()
    client.post("/profile", headers=h, json={"skin_type": "Dry", "food_habits": "dairy"})
    assert client.get("/profile", headers=h).json()["skin_type"] == "Dry"
    client.post("/profile", headers=h, json={"food_habits": ""})        # clearing now works
    p = client.get("/profile", headers=h).json()
    assert p["food_habits"] is None and p["skin_type"] == "Dry"         # untouched field kept
    assert client.post("/profile", headers=h, json={"skin_type": "<script>"}).status_code == 400


def test_chat_limits_and_offline_answer(client, make_user):
    _, h = make_user()
    r = client.post("/chat", headers=h, json={"message": "What are the symptoms of melanoma?"})
    assert r.status_code == 200 and "elanoma" in r.json()["response"]
    assert client.post("/chat", headers=h, json={"message": "x" * 5000}).status_code == 422


def test_security_headers_health_and_cors(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
    evil = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in evil.headers
    ok = client.get("/health", headers={"Origin": "http://127.0.0.1:8000"})
    assert ok.headers.get("access-control-allow-origin") == "http://127.0.0.1:8000"


def test_secret_key_is_not_the_old_hardcoded_one():
    assert app_module.settings.SECRET_KEY != "skinsense_secret_key_change_in_production"
