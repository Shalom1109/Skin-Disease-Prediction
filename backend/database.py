import datetime
import os
from pathlib import Path

from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine,
                        inspect, text)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# SQLite by default (a file next to this module, whatever directory the server starts from).
# Set SKINSENSE_DATABASE_URL=postgresql+psycopg2://user:pass@host/db to use PostgreSQL instead.
DB_PATH = os.environ.get("SKINSENSE_DB", str(Path(__file__).resolve().parent / "skinsense.db"))
SQLALCHEMY_DATABASE_URL = os.environ.get("SKINSENSE_DATABASE_URL") or f"sqlite:///{DB_PATH}"

_engine_kwargs = ({"connect_args": {"check_same_thread": False}}
                  if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {"pool_pre_ping": True})
engine = create_engine(SQLALCHEMY_DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="patient")          # "patient" or "clinician" (clinicians are set by an admin)

    skin_type = Column(String, nullable=True)
    food_habits = Column(String, nullable=True)
    products_used = Column(String, nullable=True)

    history = relationship("History", back_populates="owner", cascade="all, delete-orphan")
    lesions = relationship("Lesion", back_populates="owner", cascade="all, delete-orphan")


class Lesion(Base):
    """A spot the user tracks over time: several scans can belong to one lesion."""
    __tablename__ = "lesions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    name = Column(String)
    body_site = Column(String, nullable=True)
    created = Column(DateTime, default=_utcnow)

    owner = relationship("User", back_populates="lesions")


class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, index=True)
    prediction = Column(String)
    confidence = Column(Float)
    melanoma_prob = Column(Float, nullable=True)       # percent, kept so trends over time can be shown
    image_path = Column(String, nullable=True)         # encrypted file name inside the uploads dir
    timestamp = Column(DateTime, default=_utcnow)      # naive UTC
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    lesion_id = Column(Integer, ForeignKey("lesions.id"), nullable=True, index=True)
    age = Column(Integer, nullable=True)               # optional context supplied with the photo
    sex = Column(String, nullable=True)
    site = Column(String, nullable=True)

    owner = relationship("User", back_populates="history")
    reviews = relationship("Review", back_populates="scan", cascade="all, delete-orphan")


class Share(Base):
    """A patient's consent to let one clinician see their scans (revocable at any time)."""
    __tablename__ = "shares"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id"), index=True)
    clinician_id = Column(Integer, ForeignKey("users.id"), index=True)
    created = Column(DateTime, default=_utcnow)
    revoked_at = Column(DateTime, nullable=True)


class Review(Base):
    """A clinician's confirmation or correction of a scan (also becomes labelled data, with consent)."""
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    history_id = Column(Integer, ForeignKey("history.id"), index=True)
    clinician_id = Column(Integer, ForeignKey("users.id"), index=True)
    label = Column(String)
    note = Column(Text, nullable=True)
    created = Column(DateTime, default=_utcnow)

    scan = relationship("History", back_populates="reviews")


class AuditLog(Base):
    """Who touched whose data. Patients can read the entries about their own data."""
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, index=True)
    actor_id = Column(Integer, index=True)
    patient_id = Column(Integer, index=True)
    action = Column(String)
    detail = Column(String, nullable=True)
    created = Column(DateTime, default=_utcnow)


# Columns added after the first release; applied automatically to databases created by older versions.
_MIGRATIONS = [
    ("users", "role", "VARCHAR DEFAULT 'patient'"),
    ("history", "lesion_id", "INTEGER"),
    ("history", "melanoma_prob", "FLOAT"),
    ("history", "age", "INTEGER"),
    ("history", "sex", "VARCHAR"),
    ("history", "site", "VARCHAR"),
]


def ensure_columns(target_engine):
    """Add any missing columns to existing tables (create_all only creates whole tables)."""
    insp = inspect(target_engine)
    with target_engine.begin() as conn:
        for table, column, ddl in _MIGRATIONS:
            if table in insp.get_table_names() and column not in {c["name"] for c in insp.get_columns(table)}:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))


def init_db():
    Base.metadata.create_all(bind=engine)
    ensure_columns(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
