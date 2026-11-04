"""Care-team features: lesion tracking over time, consent-based sharing with clinicians,
clinician reviews, and an audit log the patient can read.

Rules enforced here:
* A clinician can only see a patient's data while that patient has an ACTIVE share with them.
* Every clinician access to patient data is written to the audit log, which the patient can read.
* Clinician accounts are never self-service (an admin sets the role, see scripts/manage_users.py).
"""
import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import settings
import storage
from database import AuditLog, History, Lesion, Review, Share, User, get_db

REVIEW_LABELS = settings.CLASSES + ["Other / not a skin lesion"]
BODY_SITES = ["abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital", "hand", "lower extremity",
              "neck", "scalp", "trunk", "upper extremity", "unknown"]
MAX_LESIONS = 50
ALERT_DELTA_POINTS = 15.0          # rise in melanoma probability between first and last photo that raises a flag
TRACKING_NOTE = ("Compares the model's output between photos. Differences in angle, lighting and distance affect "
                 "the result, so this does not measure how much a lesion has grown.")


# ---------------------------------------------------------------- helpers
def iso(ts):
    return ts.isoformat() + "Z" if ts.tzinfo is None else ts.isoformat()


def audit(db: Session, actor: User, patient_id: int, action: str, detail: str = ""):
    db.add(AuditLog(actor_id=actor.id, patient_id=patient_id, action=action, detail=detail[:200]))


def review_dict(r: Review, names):
    return {"id": r.id, "label": r.label, "note": r.note, "created": iso(r.created),
            "clinician": names.get(r.clinician_id, "clinician")}


def scan_dict(db: Session, h: History, image_url: Optional[str] = None):
    names = {u.id: u.username for u in db.query(User).filter(User.id.in_([r.clinician_id for r in h.reviews] or [0]))}
    return {
        "id": h.id,
        "prediction": h.prediction,
        "confidence": h.confidence,
        "melanoma_prob": h.melanoma_prob,
        "timestamp": iso(h.timestamp),
        "lesion_id": h.lesion_id,
        "age": h.age, "sex": h.sex, "site": h.site,
        "image_url": image_url if image_url is not None else (f"/history/{h.id}/image" if h.image_path else None),
        "reviews": [review_dict(r, names) for r in sorted(h.reviews, key=lambda r: r.created)],
    }


def active_share(db: Session, patient_id: int, clinician_id: int) -> Optional[Share]:
    return (db.query(Share).filter(Share.patient_id == patient_id, Share.clinician_id == clinician_id,
                                   Share.revoked_at.is_(None)).first())


def lesion_summary(db: Session, lesion: Lesion, with_scans: bool = False):
    scans = (db.query(History).filter(History.lesion_id == lesion.id).order_by(History.timestamp.asc()).all())
    out = {"id": lesion.id, "name": lesion.name, "body_site": lesion.body_site, "created": iso(lesion.created),
           "n_scans": len(scans), "note": TRACKING_NOTE, "change": None}
    if scans:
        out["last_scan"] = iso(scans[-1].timestamp)
        out["latest_prediction"] = scans[-1].prediction
    if len(scans) >= 2:
        first, last = scans[0], scans[-1]
        delta = None
        if first.melanoma_prob is not None and last.melanoma_prob is not None:
            delta = round(last.melanoma_prob - first.melanoma_prob, 1)
        became_melanoma = last.prediction.startswith("Melanoma") and not first.prediction.startswith("Melanoma")
        out["change"] = {
            "days": (last.timestamp - first.timestamp).days,
            "melanoma_prob_delta": delta,
            "first_prediction": first.prediction, "last_prediction": last.prediction,
            "alert": bool((delta is not None and delta >= ALERT_DELTA_POINTS) or became_melanoma),
        }
    if with_scans:
        out["scans"] = [scan_dict(db, s) for s in scans]
    return out


# ---------------------------------------------------------------- schemas
class LesionIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    body_site: Optional[str] = Field(default=None, max_length=40)


class AssignIn(BaseModel):
    lesion_id: Optional[int] = None


class ShareIn(BaseModel):
    clinician_username: str = Field(min_length=1, max_length=64)


class ReviewIn(BaseModel):
    label: str
    note: Optional[str] = Field(default=None, max_length=1000)


def build_router(get_current_user) -> APIRouter:
    router = APIRouter()

    def clinician_only(user: User = Depends(get_current_user)) -> User:
        if user.role != "clinician":
            raise HTTPException(403, "A clinician account is required")
        return user

    # ---------------- lesions (patient) ----------------
    def own_lesion(db: Session, user: User, lesion_id: int) -> Lesion:
        lesion = db.query(Lesion).filter(Lesion.id == lesion_id, Lesion.user_id == user.id).first()
        if not lesion:
            raise HTTPException(404, "Lesion not found")
        return lesion

    @router.get("/lesions")
    def list_lesions(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
        rows = db.query(Lesion).filter(Lesion.user_id == user.id).order_by(Lesion.created.desc()).all()
        return [lesion_summary(db, l) for l in rows]

    @router.post("/lesions")
    def create_lesion(body: LesionIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
        if body.body_site and body.body_site not in BODY_SITES:
            raise HTTPException(400, "Unknown body site")
        if db.query(Lesion).filter(Lesion.user_id == user.id).count() >= MAX_LESIONS:
            raise HTTPException(400, f"You can track at most {MAX_LESIONS} lesions")
        lesion = Lesion(user_id=user.id, name=body.name.strip(), body_site=body.body_site or None)
        db.add(lesion)
        db.commit()
        db.refresh(lesion)
        return lesion_summary(db, lesion)

    @router.get("/lesions/{lesion_id}")
    def get_lesion(lesion_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
        return lesion_summary(db, own_lesion(db, user, lesion_id), with_scans=True)

    @router.delete("/lesions/{lesion_id}")
    def delete_lesion(lesion_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
        lesion = own_lesion(db, user, lesion_id)
        db.query(History).filter(History.lesion_id == lesion.id).update({History.lesion_id: None})
        db.delete(lesion)
        db.commit()
        return {"message": "Lesion removed; its scans were kept"}

    @router.post("/history/{history_id}/lesion")
    def assign_scan(history_id: int, body: AssignIn, user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
        scan = db.query(History).filter(History.id == history_id, History.user_id == user.id).first()
        if not scan:
            raise HTTPException(404, "Scan not found")
        if body.lesion_id is not None:
            own_lesion(db, user, body.lesion_id)
        scan.lesion_id = body.lesion_id
        db.commit()
        return {"message": "Updated"}

    # ---------------- sharing (patient) ----------------
    @router.get("/sharing")
    def list_shares(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
        shares = (db.query(Share).filter(Share.patient_id == user.id, Share.revoked_at.is_(None))
                  .order_by(Share.created.desc()).all())
        names = {u.id: u.username for u in db.query(User).filter(User.id.in_([s.clinician_id for s in shares] or [0]))}
        return [{"id": s.id, "clinician": names.get(s.clinician_id), "created": iso(s.created)} for s in shares]

    @router.post("/sharing")
    def grant_share(body: ShareIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
        from sqlalchemy import func
        clin = (db.query(User).filter(func.lower(User.username) == body.clinician_username.strip().lower(),
                                      User.role == "clinician").first())
        if not clin or clin.id == user.id:
            raise HTTPException(404, "No clinician account with that username")
        share = active_share(db, user.id, clin.id)
        if not share:
            share = Share(patient_id=user.id, clinician_id=clin.id)
            db.add(share)
            audit(db, user, user.id, "share_granted", clin.username)
            db.commit()
            db.refresh(share)
        return {"id": share.id, "clinician": clin.username, "created": iso(share.created)}

    @router.delete("/sharing/{share_id}")
    def revoke_share(share_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
        share = db.query(Share).filter(Share.id == share_id, Share.patient_id == user.id,
                                       Share.revoked_at.is_(None)).first()
        if not share:
            raise HTTPException(404, "Share not found")
        share.revoked_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        clin = db.query(User).filter(User.id == share.clinician_id).first()
        audit(db, user, user.id, "share_revoked", clin.username if clin else "")
        db.commit()
        return {"message": "Access revoked"}

    @router.get("/audit")
    def my_audit(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
        rows = (db.query(AuditLog).filter(AuditLog.patient_id == user.id)
                .order_by(AuditLog.created.desc()).limit(100).all())
        names = {u.id: u.username for u in db.query(User).filter(User.id.in_([r.actor_id for r in rows] or [0]))}
        return [{"when": iso(r.created), "who": names.get(r.actor_id, "unknown"), "action": r.action,
                 "detail": r.detail} for r in rows]

    # ---------------- clinician ----------------
    def shared_patient(db: Session, clin: User, patient_id: int) -> User:
        if not active_share(db, patient_id, clin.id):
            raise HTTPException(403, "This patient has not shared their scans with you")
        patient = db.query(User).filter(User.id == patient_id).first()
        if not patient:
            raise HTTPException(404, "Patient not found")
        return patient

    @router.get("/clinician/patients")
    def clinician_patients(clin: User = Depends(clinician_only), db: Session = Depends(get_db)):
        shares = db.query(Share).filter(Share.clinician_id == clin.id, Share.revoked_at.is_(None)).all()
        out = []
        for s in shares:
            p = db.query(User).filter(User.id == s.patient_id).first()
            if not p:
                continue
            scans = db.query(History).filter(History.user_id == p.id).order_by(History.timestamp.desc()).all()
            out.append({"patient_id": p.id, "username": p.username, "shared_since": iso(s.created),
                        "n_scans": len(scans),
                        "last_scan": iso(scans[0].timestamp) if scans else None,
                        "last_prediction": scans[0].prediction if scans else None,
                        "needs_review": sum(1 for x in scans if not x.reviews)})
        return out

    @router.get("/clinician/patients/{patient_id}/history")
    def clinician_history(patient_id: int, clin: User = Depends(clinician_only), db: Session = Depends(get_db)):
        patient = shared_patient(db, clin, patient_id)
        rows = db.query(History).filter(History.user_id == patient.id).order_by(History.timestamp.desc()).all()
        audit(db, clin, patient.id, "viewed_history", f"{len(rows)} scans")
        db.commit()
        return {"patient": patient.username, "skin_type": patient.skin_type,
                "scans": [scan_dict(db, h, f"/clinician/patients/{patient.id}/history/{h.id}/image" if h.image_path else None)
                          for h in rows]}

    @router.get("/clinician/patients/{patient_id}/history/{history_id}/image")
    def clinician_image(patient_id: int, history_id: int, clin: User = Depends(clinician_only),
                        db: Session = Depends(get_db)):
        patient = shared_patient(db, clin, patient_id)
        scan = db.query(History).filter(History.id == history_id, History.user_id == patient.id).first()
        data = storage.read(scan.image_path) if scan and scan.image_path else None
        if data is None:
            raise HTTPException(404, "Image not found")
        audit(db, clin, patient.id, "viewed_image", f"scan {history_id}")
        db.commit()
        return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})

    @router.post("/clinician/scans/{history_id}/review")
    def review_scan(history_id: int, body: ReviewIn, clin: User = Depends(clinician_only),
                    db: Session = Depends(get_db)):
        if body.label not in REVIEW_LABELS:
            raise HTTPException(400, "Unknown label")
        scan = db.query(History).filter(History.id == history_id).first()
        if not scan or not active_share(db, scan.user_id, clin.id):
            raise HTTPException(403, "This patient has not shared that scan with you")
        review = db.query(Review).filter(Review.history_id == scan.id, Review.clinician_id == clin.id).first()
        if review is None:
            review = Review(history_id=scan.id, clinician_id=clin.id)
            db.add(review)
        review.label, review.note = body.label, (body.note or None)
        audit(db, clin, scan.user_id, "reviewed_scan", f"scan {scan.id}: {body.label}")
        db.commit()
        db.refresh(review)
        return review_dict(review, {clin.id: clin.username})

    return router
