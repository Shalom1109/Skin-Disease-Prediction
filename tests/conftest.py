"""Test setup: isolated DB / uploads / secret, no weights or network needed."""
import io
import os
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))              # `import backend.model`, `import backend.metrics`
sys.path.insert(0, str(ROOT / "backend"))  # app modules import each other as top-level names

_TMP = tempfile.mkdtemp(prefix="skinsense-tests-")
os.environ["SKINSENSE_DB"] = os.path.join(_TMP, "test.db")
os.environ["SKINSENSE_UPLOADS"] = os.path.join(_TMP, "uploads")
os.environ["SKINSENSE_SECRET_KEY"] = "test-secret-key-not-for-production"
from cryptography.fernet import Fernet  # noqa: E402
os.environ["SKINSENSE_IMAGE_KEY"] = Fernet.generate_key().decode()
os.environ["SKINSENSE_CALIBRATION"] = os.path.join(_TMP, "no-calibration.json")
os.environ.pop("SKINSENSE_LLM", None)
os.environ.pop("SKINSENSE_LLM_URL", None)

import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

import app as app_module  # noqa: E402

NEVI = [0, 0, 0, 0, 0, 5.0, 0]            # confident nevus, P(mel) ~ 0.6 %
NEVI_BUT_MEL_RISK = [0, 0, 0, 0, 3.0, 5.0, 0]  # nevus on top, P(mel) ~ 11 %
UNSURE = [0.5, 0.4, 0.45, 0.4, 0.5, 0.55, 0.4]  # flat -> top prob ~ 15 %


class FakeModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor([logits], dtype=torch.float32)

    def forward(self, x):
        return self.logits.clone()


class FakeStore:
    def __init__(self, logits=NEVI):
        self.model = FakeModel(logits)
        self.loaded = True
        self.checkpoint = "fake"

    def get(self):
        return self.model


class MissingStore:
    loaded = False

    def get(self):
        raise app_module.ModelUnavailable("No trained weights found.")


@pytest.fixture(autouse=True)
def _reset_state():
    app_module.login_limiter._events.clear()
    app_module.register_limiter._events.clear()
    yield


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(app_module, "model_store", FakeStore())
    return TestClient(app_module.app)


@pytest.fixture()
def make_user(client):
    def _make(password="Passw0rd!x"):
        name = f"user_{uuid.uuid4().hex[:10]}"
        r = client.post("/register", data={"username": name, "password": password})
        assert r.status_code == 200, r.text
        token = client.post("/login", data={"username": name, "password": password}).json()["access_token"]
        return name, {"Authorization": f"Bearer {token}"}

    return _make


def image_bytes(fmt="JPEG", size=(64, 64), exif=False, color=(180, 120, 90)):
    buf = io.BytesIO()
    img = Image.new("RGB", size, color)
    kwargs = {}
    if exif:
        e = Image.Exif()
        e[0x010F] = "SecretCameraMaker"  # Make
        kwargs["exif"] = e
    img.save(buf, fmt, **kwargs)
    return buf.getvalue()
