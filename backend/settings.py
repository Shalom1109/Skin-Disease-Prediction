"""Central configuration for the SkinSense backend.

Everything that used to be hardcoded in app.py (secret key, CORS, limits,
calibration constants) lives here and can be overridden with environment
variables.
"""
import json
import logging
import os
import secrets
from pathlib import Path

logger = logging.getLogger("skinsense")

BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
KB_PATH = ROOT_DIR / "data" / "knowledge_base.json"
UPLOADS_DIR = Path(os.environ.get("SKINSENSE_UPLOADS", BACKEND_DIR / "uploads"))
CALIBRATION_PATH = Path(os.environ.get("SKINSENSE_CALIBRATION", BACKEND_DIR / "calibration.json"))

# Class order MUST match the ImageFolder (alphabetical) order used in training:
# akiec, bcc, bkl, df, mel, nv, vasc
CLASSES = [
    "Actinic Keratoses",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Nevi",
    "Vascular",
]
NUM_CLASSES = len(CLASSES)
MELANOMA = "Melanoma"
ESCALATED_LABEL = "Melanoma (High-Priority Escalation)"
UNCERTAIN_LABEL = "Uncertain / Non-Skin Image"

# --- Auth -------------------------------------------------------------------
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("SKINSENSE_TOKEN_MINUTES", "60"))
MIN_PASSWORD_LENGTH = 8
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_SECONDS = 300
REGISTER_MAX_PER_HOUR = 10


def _load_secret_key() -> str:
    """Secret comes from the environment, else a random per-install key that is
    generated once and stored (git-ignored) in backend/.secret_key."""
    env_key = os.environ.get("SKINSENSE_SECRET_KEY")
    if env_key:
        return env_key
    key_file = BACKEND_DIR / ".secret_key"
    if key_file.exists():
        return key_file.read_text().strip()
    key = secrets.token_urlsafe(48)
    key_file.write_text(key)
    try:
        os.chmod(key_file, 0o600)
    except OSError:
        pass
    logger.info("Generated a new secret key at %s", key_file)
    return key


SECRET_KEY = _load_secret_key()

# --- HTTP -------------------------------------------------------------------
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "SKINSENSE_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000"
    ).split(",")
    if o.strip()
]

# --- Uploads ----------------------------------------------------------------
MAX_UPLOAD_BYTES = int(os.environ.get("SKINSENSE_MAX_UPLOAD_MB", "8")) * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
STORED_IMAGE_MAX_SIDE = 1600

# --- Chat -------------------------------------------------------------------
MAX_CHAT_MESSAGE_CHARS = 1000

# --- Model ------------------------------------------------------------------
# By default the API refuses to predict without trained weights instead of
# silently returning random-weight output.
ALLOW_UNTRAINED = os.environ.get("SKINSENSE_ALLOW_UNTRAINED") == "1"

DEFAULT_CALIBRATION = {
    "temperature": 1.0,
    # Escalate to "Melanoma" when P(melanoma) >= this and it is not already the
    # top class. 0.04 is the legacy hand-picked value; run evaluate.py to fit a
    # validated one (target sensitivity on the validation split).
    "melanoma_threshold": 0.04,
    # Below this top-class probability the image is reported as uncertain.
    "min_confidence": 0.35,
    "source": "default (not validated - run evaluate.py)",
}


def load_calibration() -> dict:
    cfg = dict(DEFAULT_CALIBRATION)
    if CALIBRATION_PATH.exists():
        try:
            with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
                cfg.update({k: v for k, v in json.load(f).items() if k in DEFAULT_CALIBRATION or k == "metrics"})
        except (OSError, ValueError) as exc:
            logger.warning("Could not read %s (%s); using defaults.", CALIBRATION_PATH, exc)
    cfg["temperature"] = max(float(cfg["temperature"]), 1e-3)
    return cfg
