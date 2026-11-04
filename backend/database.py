import datetime
import os
from pathlib import Path

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# Always resolve relative to this file so the same database is used no matter
# which directory the server is launched from.
DB_PATH = os.environ.get("SKINSENSE_DB", str(Path(__file__).resolve().parent / "skinsense.db"))
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

    skin_type = Column(String, nullable=True)
    food_habits = Column(String, nullable=True)
    products_used = Column(String, nullable=True)

    history = relationship("History", back_populates="owner", cascade="all, delete-orphan")


class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, index=True)
    prediction = Column(String)
    confidence = Column(Float)
    image_path = Column(String, nullable=True)  # file name inside the uploads dir
    timestamp = Column(DateTime, default=_utcnow)  # naive UTC
    user_id = Column(Integer, ForeignKey("users.id"), index=True)

    owner = relationship("User", back_populates="history")


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
