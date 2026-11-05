import datetime
import io
import logging
import os
import re
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from passlib.context import CryptContext
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

import settings
from database import AuditLog, History, Lesion, Review, Share, User, get_db, init_db
from inference import decide, predict_probs, preprocess, referral
from fusion import MetadataFusion
from quality import assess_quality
import storage
from care import BODY_SITES, active_share, audit, build_router, scan_dict
from explain import ensemble_cam, overlay_png
from rag import SkinSenseRAG
from security import RateLimiter

logging.basicConfig(level=os.environ.get("SKINSENSE_LOG", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("skinsense")

DISCLAIMER = (
    "SkinSense AI is an educational screening aid, not a medical device. It cannot diagnose "
    "skin conditions and was trained on dermoscopic images (HAM10000), so results on phone "
    "photos or darker skin tones may be unreliable. Always consult a qualified dermatologist."
)
SKIN_TYPES = {"", "Oily", "Dry", "Combination", "Sensitive", "Normal"}

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
# Verifying against this when the user does not exist keeps timing similar.
_DUMMY_HASH = pwd_context.hash("not-a-real-password")

login_limiter = RateLimiter(settings.LOGIN_MAX_FAILURES, settings.LOGIN_WINDOW_SECONDS)
register_limiter = RateLimiter(settings.REGISTER_MAX_PER_HOUR, 3600)

init_db()
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
calibration = settings.load_calibration()
rag = SkinSenseRAG(kb_path=settings.KB_PATH)
# Optional: combine the image model with age/sex/body site (only if scripts/train_fusion.py showed a gain).
fusion = MetadataFusion.load(os.environ.get("SKINSENSE_FUSION", settings.BACKEND_DIR / "fusion.json"))


# --------------------------------------------------------------------------- model
class ModelUnavailable(RuntimeError):
    pass


def find_checkpoint():
    """Prefer an explicit env var, then '*best*', then '*final*'. Training
    checkpoints (checkpoint_latest.pth) are never picked implicitly."""
    explicit = os.environ.get("SKINSENSE_WEIGHTS")
    if explicit:
        return explicit if os.path.exists(explicit) else None
    candidates = [f for f in os.listdir(settings.BACKEND_DIR)
                  if f.endswith(".pth") and "checkpoint" not in f.lower()]
    for keyword in ("best", "final"):
        for name in candidates:
            if keyword in name.lower():
                return str(settings.BACKEND_DIR / name)
    return None


class ModelStore:
    """Lazy, thread-safe model holder so importing the app never needs weights."""

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self.checkpoint = None

    @property
    def loaded(self):
        return self._model is not None

    def get(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load()
        return self._model

    def _load(self):
        from model import get_model

        self.checkpoint = find_checkpoint()
        if self.checkpoint is None and not settings.ALLOW_UNTRAINED:
            raise ModelUnavailable(
                "No trained weights found. Put skin_sense_ensemble_best.pth in backend/ "
                "(or set SKINSENSE_WEIGHTS).")
        logger.info("Loading model weights from %s ...", self.checkpoint or "<random init>")
        model = get_model(num_classes=settings.NUM_CLASSES, checkpoint_path=self.checkpoint)
        import torch
        use_gpu = torch.cuda.is_available() and os.environ.get("SKINSENSE_DEVICE", "auto").lower() != "cpu"
        model = model.to("cuda" if use_gpu else "cpu")
        logger.info("Model loaded and ready for inference on %s.", "GPU" if use_gpu else "CPU")
        return model


model_store = ModelStore()
_cam_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app):
    try:
        model_store.get()
    except ModelUnavailable as exc:
        logger.warning("%s The server will start, but /predict returns 503 until weights exist.", exc)
    yield


app = FastAPI(title="SkinSense AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "geolocation=(self), camera=(), microphone=()")
    return response


app.mount("/css", StaticFiles(directory=settings.FRONTEND_DIR / "css"), name="css")
app.mount("/js", StaticFiles(directory=settings.FRONTEND_DIR / "js"), name="js")
app.mount("/img", StaticFiles(directory=settings.FRONTEND_DIR / "img"), name="img")
# NOTE: uploads are deliberately NOT mounted as public static files; they are
# served per-user through GET /history/{id}/image.


# --------------------------------------------------------------------------- auth
def create_access_token(data: dict):
    to_encode = data.copy()
    to_encode["exp"] = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        raise credentials_exception
    username = payload.get("sub")
    if not username:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


def client_ip(request: Request):
    return request.client.host if request.client else "unknown"


@app.get("/")
def read_root():
    return FileResponse(settings.FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_store.loaded,
        "calibration_source": calibration.get("source"),
        "context_fusion": fusion is not None,
    }


@app.post("/register")
def register(request: Request, username: str = Form(...), password: str = Form(...),
             db: Session = Depends(get_db)):
    ip = client_ip(request)
    wait = register_limiter.retry_after(ip)
    if wait:
        raise HTTPException(429, "Too many registrations. Try again later.", headers={"Retry-After": str(wait)})

    if not re.fullmatch(r"[A-Za-z0-9_]{3,32}", username):
        raise HTTPException(400, "Username must be 3-32 characters: letters, numbers or underscores.")
    if len(password) < settings.MIN_PASSWORD_LENGTH:
        raise HTTPException(400, f"Password must be at least {settings.MIN_PASSWORD_LENGTH} characters.")
    if not re.search(r"[a-z]", password):
        raise HTTPException(400, "Password must contain at least one lowercase letter.")
    if not re.search(r"\d", password):
        raise HTTPException(400, "Password must contain at least one number.")
    if not re.search(r"[@_!#$%^&*()<>?/\\|}{~:]", password):
        raise HTTPException(400, "Password must contain at least one special character.")

    if db.query(User).filter(func.lower(User.username) == username.lower()).first():
        raise HTTPException(400, "Username already registered")

    register_limiter.record(ip)
    db.add(User(username=username, hashed_password=pwd_context.hash(password)))
    db.commit()
    return {"message": "User registered successfully"}


@app.post("/login")
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    key = f"{client_ip(request)}:{form_data.username.lower()}"
    wait = login_limiter.retry_after(key)
    if wait:
        raise HTTPException(429, "Too many failed attempts. Try again later.", headers={"Retry-After": str(wait)})

    user = db.query(User).filter(func.lower(User.username) == form_data.username.lower()).first()
    # One generic error for "no such user" and "wrong password" (no username enumeration).
    valid = pwd_context.verify(form_data.password, user.hashed_password if user else _DUMMY_HASH)
    if not user or not valid:
        login_limiter.record(key)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid username or password",
                            headers={"WWW-Authenticate": "Bearer"})

    login_limiter.reset(key)
    return {"access_token": create_access_token({"sub": user.username}), "token_type": "bearer"}


# --------------------------------------------------------------------------- info
@app.get("/diseases")
def get_diseases():
    return {k: v for k, v in rag.knowledge_base.items() if k in settings.CLASSES}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=settings.MAX_CHAT_MESSAGE_CHARS)
    context: str = Field(default="", max_length=100)


@app.post("/chat")
def chat_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    return {"response": rag.chat(request.message, request.context)}


# --------------------------------------------------------------------------- history
def _iso_utc(ts):
    return ts.isoformat() + "Z" if ts.tzinfo is None else ts.isoformat()


@app.get("/history")
def get_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (db.query(History).filter(History.user_id == current_user.id)
            .order_by(History.timestamp.desc()).all())
    return [scan_dict(db, h) for h in rows]


@app.get("/history/{history_id}/image")
def history_image(history_id: int, current_user: User = Depends(get_current_user),
                  db: Session = Depends(get_db)):
    row = db.query(History).filter(History.id == history_id, History.user_id == current_user.id).first()
    data = storage.read(row.image_path) if row and row.image_path else None
    if data is None:
        raise HTTPException(404, "Image not found")
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


# --------------------------------------------------------------------------- heat-maps
def _target_class(prediction: str, probs=None) -> int:
    """Class the heat-map explains: the one that was reported to the user."""
    if prediction.startswith("Melanoma"):
        return settings.CLASSES.index(settings.MELANOMA)
    if prediction in settings.CLASSES:
        return settings.CLASSES.index(prediction)
    return int(probs.argmax())


def _heatmap_png(scan: History) -> bytes:
    cache = storage.derived_name(scan.image_path, "cam")
    cached = storage.read(cache)
    if cached:
        return cached
    data = storage.read(scan.image_path)
    if data is None:
        raise HTTPException(404, "Image not found")
    try:
        model = model_store.get()
    except ModelUnavailable as exc:
        raise HTTPException(503, str(exc))
    image = Image.open(io.BytesIO(data)).convert("RGB")
    tensor = preprocess(image)
    try:
        probs = None if (scan.prediction.startswith("Melanoma") or scan.prediction in settings.CLASSES) \
            else predict_probs(model, tensor, calibration["temperature"])
        with _cam_lock:
            cam = ensemble_cam(model, tensor, _target_class(scan.prediction, probs))
        png = overlay_png(image, cam)
    except Exception:
        logger.exception("Heat-map failed")
        raise HTTPException(500, "The heat-map could not be generated.")
    storage.write(cache, png)
    return png


@app.get("/history/{history_id}/heatmap")
def history_heatmap(history_id: int, current_user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    row = db.query(History).filter(History.id == history_id, History.user_id == current_user.id).first()
    if not row or not row.image_path:
        raise HTTPException(404, "Image not found")
    return Response(content=_heatmap_png(row), media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/clinician/patients/{patient_id}/history/{history_id}/heatmap")
def clinician_heatmap(patient_id: int, history_id: int, current_user: User = Depends(get_current_user),
                      db: Session = Depends(get_db)):
    if current_user.role != "clinician":
        raise HTTPException(403, "A clinician account is required")
    if not active_share(db, patient_id, current_user.id):
        raise HTTPException(403, "This patient has not shared their scans with you")
    row = db.query(History).filter(History.id == history_id, History.user_id == patient_id).first()
    if not row or not row.image_path:
        raise HTTPException(404, "Image not found")
    png = _heatmap_png(row)
    audit(db, current_user, patient_id, "viewed_heatmap", f"scan {history_id}")
    db.commit()
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "private, no-store"})


def _delete_image(name):
    storage.delete(name, "cam")


@app.delete("/history/{history_id}")
def delete_history_item(history_id: int, current_user: User = Depends(get_current_user),
                        db: Session = Depends(get_db)):
    row = db.query(History).filter(History.id == history_id, History.user_id == current_user.id).first()
    if not row:
        raise HTTPException(404, "Not found")
    _delete_image(row.image_path)
    db.delete(row)
    db.commit()
    return {"message": "Deleted"}


@app.delete("/account")
def delete_account(password: str = Form(...), current_user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    """Right to erasure: removes the user, every history row and every stored image."""
    if not pwd_context.verify(password, current_user.hashed_password):
        raise HTTPException(401, "Incorrect password")
    for row in current_user.history:
        _delete_image(row.image_path)
    uid = current_user.id
    db.query(Share).filter((Share.patient_id == uid) | (Share.clinician_id == uid)).delete(synchronize_session=False)
    db.query(AuditLog).filter((AuditLog.patient_id == uid) | (AuditLog.actor_id == uid)).delete(synchronize_session=False)
    db.query(Review).filter(Review.clinician_id == uid).delete(synchronize_session=False)
    db.expire_all()
    db.delete(current_user)
    db.commit()
    return {"message": "Account and all stored data deleted"}


# --------------------------------------------------------------------------- profile
class ProfileUpdate(BaseModel):
    skin_type: Optional[str] = Field(default=None, max_length=20)
    food_habits: Optional[str] = Field(default=None, max_length=500)
    products_used: Optional[str] = Field(default=None, max_length=500)


@app.get("/profile")
def get_profile(current_user: User = Depends(get_current_user)):
    return {
        "username": current_user.username,
        "role": current_user.role or "patient",
        "skin_type": current_user.skin_type,
        "food_habits": current_user.food_habits,
        "products_used": current_user.products_used,
    }


@app.post("/profile")
def update_profile(profile: ProfileUpdate, current_user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    # Only fields the client actually sent are touched, so "" can clear a value.
    sent = profile.model_fields_set
    if "skin_type" in sent:
        if profile.skin_type not in SKIN_TYPES and profile.skin_type is not None:
            raise HTTPException(400, "Invalid skin type")
        current_user.skin_type = profile.skin_type or None
    if "food_habits" in sent:
        current_user.food_habits = profile.food_habits or None
    if "products_used" in sent:
        current_user.products_used = profile.products_used or None
    db.commit()
    return {"message": "Profile updated successfully"}


# --------------------------------------------------------------------------- predict
UNCERTAIN_INFO = {
    "description": "The AI could not confidently identify a recognizable skin lesion in this image. This usually happens if a non-skin image is uploaded, or if the photo is blurry or poorly lit.",
    "symptoms": ["N/A"],
    "recommendations": [
        "Upload a clear, well-lit photograph focused on the skin.",
        "If you are worried about a skin change, consult a dermatologist regardless of this result.",
    ],
    "dos": ["Take the photo in natural lighting.", "Keep the camera steady and focused."],
    "donts": ["Upload pictures of pets, furniture or other objects.", "Upload blurry or very dark photos."],
}


def _read_image(file: UploadFile) -> Image.Image:
    data = file.file.read(settings.MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image too large (max {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
    try:
        probe = Image.open(io.BytesIO(data))
        if probe.format not in settings.ALLOWED_IMAGE_FORMATS:
            raise HTTPException(415, "Unsupported image type. Use JPEG, PNG or WebP.")
        if probe.width * probe.height > settings.MAX_IMAGE_PIXELS:
            raise HTTPException(400, "Image dimensions are too large")
        image = Image.open(io.BytesIO(data))
        image.load()
    except HTTPException:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        raise HTTPException(400, "File is not a valid image")
    # Apply EXIF rotation, then re-encode later: the saved copy has no EXIF/GPS data.
    return ImageOps.exif_transpose(image).convert("RGB")


def _store_image(image: Image.Image) -> str:
    stored = image.copy()
    stored.thumbnail((settings.STORED_IMAGE_MAX_SIDE, settings.STORED_IMAGE_MAX_SIDE))
    buffer = io.BytesIO()
    stored.save(buffer, "JPEG", quality=92)          # re-encoding drops EXIF/GPS metadata
    return storage.save(buffer.getvalue())            # encrypted before it reaches the disk


@app.post("/quality")
def check_quality(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    """Instant photo-quality feedback before running the analysis (no model involved)."""
    return assess_quality(_read_image(file))


# Plain `def` (not `async def`): FastAPI runs it in a worker thread, so heavy
# CPU-bound inference no longer blocks the event loop for other requests.
@app.post("/predict")
def predict(file: UploadFile = File(...), lesion_id: Optional[int] = Form(None),
            age: Optional[int] = Form(None), sex: Optional[str] = Form(None), site: Optional[str] = Form(None),
            current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if lesion_id is not None and not db.query(Lesion).filter(
            Lesion.id == lesion_id, Lesion.user_id == current_user.id).first():
        raise HTTPException(404, "Lesion not found")
    if age is not None and not 0 <= age <= 120:
        raise HTTPException(400, "Age must be between 0 and 120")
    if sex not in (None, "", "male", "female"):
        raise HTTPException(400, "Sex must be 'male' or 'female'")
    if site not in (None, "") and site not in BODY_SITES:
        raise HTTPException(400, "Unknown body site")
    image = _read_image(file)
    quality = assess_quality(image)

    try:
        model = model_store.get()
    except ModelUnavailable as exc:
        raise HTTPException(503, str(exc))

    try:
        probs = predict_probs(model, preprocess(image), calibration["temperature"])
        cfg, context_used = calibration, False
        if fusion is not None and MetadataFusion.has_context(age, sex, site):
            probs = fusion.predict(probs, age, sex, site)
            cfg = {**calibration, "melanoma_threshold": fusion.melanoma_threshold}
            context_used = True
        result = decide(probs, settings.CLASSES, cfg, settings.MELANOMA,
                        settings.ESCALATED_LABEL, settings.UNCERTAIN_LABEL)
        info = UNCERTAIN_INFO if result["class_name"] is None else rag.retrieve_info(result["class_name"])
        image_name = _store_image(image)
        scan = History(prediction=result["prediction"], confidence=result["confidence"],
                       melanoma_prob=result["melanoma_probability"], image_path=image_name,
                       user_id=current_user.id, lesion_id=lesion_id,
                       age=age, sex=sex or None, site=site or None)
        db.add(scan)
        db.commit()
        db.refresh(scan)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(500, "Prediction failed. Please try again with a different image.")

    return {
        "scan_id": scan.id,
        "context_used": context_used,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "escalated": result["escalated"],
        "uncertain": result["uncertain"],
        "note": result["note"],
        "top_prediction": result["top_prediction"],
        "melanoma_probability": result["melanoma_probability"],
        "probabilities": result["probabilities"],
        "description": info["description"],
        "symptoms": info["symptoms"],
        "recommendations": info["recommendations"],
        "dos": info["dos"],
        "donts": info["donts"],
        "referral": referral(result, quality),
        "quality": quality,
        "disclaimer": DISCLAIMER,
    }


app.include_router(build_router(get_current_user))


@app.exception_handler(Exception)
async def unhandled(_request: Request, exc: Exception):
    logger.exception("Unhandled error", exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
