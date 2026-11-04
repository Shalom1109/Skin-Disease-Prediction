import os
import uuid

from sqlalchemy import create_engine, text

import app as app_module
import database
from conftest import NEVI, NEVI_BUT_MEL_RISK, FakeStore, image_bytes


def upload(client, headers, **extra):
    files = {"file": ("lesion.jpg", image_bytes(), "image/jpeg")}
    return client.post("/predict", headers=headers, files=files, data=extra or None)


def make_clinician(client, make_user):
    name, headers = make_user()
    with database.SessionLocal() as db:
        db.query(database.User).filter(database.User.username == name).update({"role": "clinician"})
        db.commit()
    return name, headers


# ---------------------------------------------------------------- schema migration / roles
def test_old_databases_are_migrated_in_place(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with engine.begin() as c:                       # the schema shipped in the very first release
        c.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, username VARCHAR, hashed_password VARCHAR, "
                       "skin_type VARCHAR, food_habits VARCHAR, products_used VARCHAR)"))
        c.execute(text("CREATE TABLE history (id INTEGER PRIMARY KEY, prediction VARCHAR, confidence FLOAT, "
                       "image_path VARCHAR, timestamp DATETIME, user_id INTEGER)"))
        c.execute(text("INSERT INTO users (id, username, hashed_password) VALUES (1, 'old_user', 'x')"))
    database.ensure_columns(engine)
    with engine.connect() as c:
        cols = {r[1] for r in c.execute(text("PRAGMA table_info(history)"))}
        assert {"lesion_id", "melanoma_prob", "age", "sex", "site"} <= cols
        assert c.execute(text("SELECT role FROM users WHERE id = 1")).scalar() == "patient"
    database.ensure_columns(engine)                  # running it twice is harmless


def test_new_accounts_are_patients_and_cannot_use_clinician_endpoints(client, make_user):
    _, h = make_user()
    assert client.get("/profile", headers=h).json()["role"] == "patient"
    assert client.get("/clinician/patients", headers=h).status_code == 403
    assert client.post("/clinician/scans/1/review", headers=h, json={"label": "Nevi"}).status_code == 403


def test_registration_cannot_create_a_clinician(client):
    r = client.post("/register", data={"username": "sneaky_doc", "password": "Passw0rd!x", "role": "clinician"})
    assert r.status_code == 200
    token = client.post("/login", data={"username": "sneaky_doc", "password": "Passw0rd!x"}).json()["access_token"]
    assert client.get("/profile", headers={"Authorization": f"Bearer {token}"}).json()["role"] == "patient"


# ---------------------------------------------------------------- consent-based sharing
def test_clinician_sees_nothing_until_the_patient_shares(client, make_user):
    clin_name, clin = make_clinician(client, make_user)
    _, pat = make_user()
    upload(client, pat)
    patient_id = client.get("/history", headers=pat).json()[0]["id"]
    assert client.get("/clinician/patients", headers=clin).json() == []
    # guessing ids does not work either
    assert client.get("/clinician/patients/1/history", headers=clin).status_code == 403
    assert client.post(f"/clinician/scans/{patient_id}/review", headers=clin, json={"label": "Nevi"}).status_code == 403


def test_full_sharing_flow_with_audit_trail_and_revocation(client, make_user):
    clin_name, clin = make_clinician(client, make_user)
    pat_name, pat = make_user()
    upload(client, pat)

    share = client.post("/sharing", headers=pat, json={"clinician_username": clin_name.upper()}).json()
    assert share["clinician"] == clin_name
    again = client.post("/sharing", headers=pat, json={"clinician_username": clin_name}).json()
    assert again["id"] == share["id"]                                      # idempotent

    patients = client.get("/clinician/patients", headers=clin).json()
    assert [p["username"] for p in patients] == [pat_name] and patients[0]["needs_review"] == 1
    pid = patients[0]["patient_id"]

    hist = client.get(f"/clinician/patients/{pid}/history", headers=clin).json()
    assert hist["patient"] == pat_name and len(hist["scans"]) == 1
    img = client.get(hist["scans"][0]["image_url"], headers=clin)
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"

    audit = client.get("/audit", headers=pat).json()
    actions = [a["action"] for a in audit]
    assert {"share_granted", "viewed_history", "viewed_image"} <= set(actions)
    assert all(a["who"] in (clin_name, pat_name) for a in audit)

    assert client.delete(f"/sharing/{share['id']}", headers=pat).status_code == 200
    assert client.get("/clinician/patients", headers=clin).json() == []
    assert client.get(f"/clinician/patients/{pid}/history", headers=clin).status_code == 403
    assert client.get(hist["scans"][0]["image_url"], headers=clin).status_code == 403
    assert "share_revoked" in [a["action"] for a in client.get("/audit", headers=pat).json()]


def test_you_can_only_share_with_real_clinicians_not_yourself_or_other_patients(client, make_user):
    pat_name, pat = make_user()
    other_name, _ = make_user()
    assert client.post("/sharing", headers=pat, json={"clinician_username": other_name}).status_code == 404
    assert client.post("/sharing", headers=pat, json={"clinician_username": "nobody_here"}).status_code == 404
    assert client.post("/sharing", headers=pat, json={"clinician_username": pat_name}).status_code == 404


def test_one_patient_cannot_read_or_revoke_anothers_shares(client, make_user):
    clin_name, _ = make_clinician(client, make_user)
    _, alice = make_user()
    _, bob = make_user()
    share = client.post("/sharing", headers=alice, json={"clinician_username": clin_name}).json()
    assert client.get("/sharing", headers=bob).json() == []
    assert client.delete(f"/sharing/{share['id']}", headers=bob).status_code == 404
    assert client.get("/audit", headers=bob).json() == []


# ---------------------------------------------------------------- clinician review
def test_clinician_review_is_visible_to_the_patient_and_can_be_updated(client, make_user):
    clin_name, clin = make_clinician(client, make_user)
    _, pat = make_user()
    upload(client, pat)
    scan_id = client.get("/history", headers=pat).json()[0]["id"]
    client.post("/sharing", headers=pat, json={"clinician_username": clin_name})

    assert client.post(f"/clinician/scans/{scan_id}/review", headers=clin, json={"label": "Not a label"}).status_code == 400
    r = client.post(f"/clinician/scans/{scan_id}/review", headers=clin, json={"label": "Nevi", "note": "benign mole"})
    assert r.status_code == 200 and r.json()["label"] == "Nevi"
    client.post(f"/clinician/scans/{scan_id}/review", headers=clin, json={"label": "Melanoma", "note": "biopsy: positive"})

    reviews = client.get("/history", headers=pat).json()[0]["reviews"]
    assert len(reviews) == 1 and reviews[0]["label"] == "Melanoma" and reviews[0]["clinician"] == clin_name
    assert client.get("/clinician/patients", headers=clin).json()[0]["needs_review"] == 0


# ---------------------------------------------------------------- lesion tracking
def test_lesion_tracking_shows_change_and_alerts_when_the_risk_rises(client, make_user, monkeypatch):
    _, h = make_user()
    lesion = client.post("/lesions", headers=h, json={"name": "mole on left arm", "body_site": "upper extremity"}).json()
    monkeypatch.setattr(app_module, "model_store", FakeStore(NEVI))
    assert upload(client, h, lesion_id=str(lesion["id"])).status_code == 200
    monkeypatch.setattr(app_module, "model_store", FakeStore([0, 0, 0, 0, 3.9, 5.0, 0]))
    assert upload(client, h, lesion_id=str(lesion["id"])).status_code == 200

    detail = client.get(f"/lesions/{lesion['id']}", headers=h).json()
    assert detail["n_scans"] == 2 and len(detail["scans"]) == 2
    change = detail["change"]
    assert change["melanoma_prob_delta"] > 10 and change["alert"] is True
    assert "does not measure" in detail["note"]
    assert client.get("/lesions", headers=h).json()[0]["n_scans"] == 2


def test_stable_lesion_does_not_alert(client, make_user, monkeypatch):
    _, h = make_user()
    lesion = client.post("/lesions", headers=h, json={"name": "stable spot"}).json()
    monkeypatch.setattr(app_module, "model_store", FakeStore(NEVI))
    upload(client, h, lesion_id=str(lesion["id"]))
    upload(client, h, lesion_id=str(lesion["id"]))
    assert client.get(f"/lesions/{lesion['id']}", headers=h).json()["change"]["alert"] is False


def test_lesions_are_private_and_validated(client, make_user):
    _, alice = make_user()
    _, bob = make_user()
    lesion = client.post("/lesions", headers=alice, json={"name": "mine"}).json()
    assert client.get(f"/lesions/{lesion['id']}", headers=bob).status_code == 404
    assert client.delete(f"/lesions/{lesion['id']}", headers=bob).status_code == 404
    assert upload(client, bob, lesion_id=str(lesion["id"])).status_code == 404     # cannot attach scans to it
    assert client.post("/lesions", headers=alice, json={"name": "x", "body_site": "elbow?"}).status_code == 400
    assert client.post("/lesions", headers=alice, json={"name": ""}).status_code == 422


def test_assigning_and_removing_a_lesion_keeps_the_scans(client, make_user):
    _, h = make_user()
    upload(client, h)
    scan = client.get("/history", headers=h).json()[0]
    lesion = client.post("/lesions", headers=h, json={"name": "spot"}).json()
    assert client.post(f"/history/{scan['id']}/lesion", headers=h, json={"lesion_id": lesion["id"]}).status_code == 200
    assert client.get("/history", headers=h).json()[0]["lesion_id"] == lesion["id"]
    assert client.delete(f"/lesions/{lesion['id']}", headers=h).status_code == 200
    after = client.get("/history", headers=h).json()
    assert len(after) == 1 and after[0]["lesion_id"] is None


# ---------------------------------------------------------------- optional context
def test_optional_context_is_stored_and_validated(client, make_user):
    _, h = make_user()
    assert upload(client, h, age="54", sex="female", site="back").status_code == 200
    scan = client.get("/history", headers=h).json()[0]
    assert (scan["age"], scan["sex"], scan["site"]) == (54, "female", "back")
    assert upload(client, h, age="200").status_code == 400
    assert upload(client, h, sex="robot").status_code == 400
    assert upload(client, h, site="elbow?").status_code == 400


# ---------------------------------------------------------------- erasure covers the new data too
def test_deleting_an_account_removes_shares_audit_lesions_and_reviews(client, make_user):
    clin_name, clin = make_clinician(client, make_user)
    pat_name, pat = make_user()
    upload(client, pat)
    scan_id = client.get("/history", headers=pat).json()[0]["id"]
    client.post("/lesions", headers=pat, json={"name": "spot"})
    client.post("/sharing", headers=pat, json={"clinician_username": clin_name})
    client.post(f"/clinician/scans/{scan_id}/review", headers=clin, json={"label": "Nevi"})

    r = client.request("DELETE", "/account", headers=pat, data={"password": "Passw0rd!x"})
    assert r.status_code == 200
    assert client.get("/clinician/patients", headers=clin).json() == []
    with database.SessionLocal() as db:
        assert db.query(database.Lesion).filter(database.Lesion.name == "spot").count() == 0
        assert db.query(database.Review).filter(database.Review.history_id == scan_id).count() == 0
        assert db.query(database.AuditLog).filter(database.AuditLog.detail == clin_name).count() == 0
