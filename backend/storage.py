"""Encrypted-at-rest storage for uploaded photos.

Every new image is encrypted with Fernet (AES-128-CBC + HMAC) before it touches the disk, so a stolen disk or
backup does not expose patient photos. The key comes from SKINSENSE_IMAGE_KEY or is generated once into
backend/.image_key (git-ignored). LOSING THE KEY MEANS LOSING THE IMAGES - back it up separately.
Legacy plain `.jpg` files written by older versions can still be read and deleted.
"""
import os
import uuid
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

import settings


def _load_key() -> bytes:
    env = os.environ.get("SKINSENSE_IMAGE_KEY")
    if env:
        return env.encode()
    key_file = settings.BACKEND_DIR / ".image_key"
    if key_file.exists():
        return key_file.read_text().strip().encode()
    key = Fernet.generate_key()
    key_file.write_bytes(key)
    try:
        os.chmod(key_file, 0o600)
    except OSError:
        pass
    return key


_fernet = Fernet(_load_key())


def _path(name: str) -> Optional[Path]:
    root = settings.UPLOADS_DIR.resolve()
    path = (root / name).resolve()
    return path if path.parent == root else None          # blocks '../' tricks


def save(data: bytes) -> str:
    """Encrypt and store `data`; returns the new file name."""
    name = f"{uuid.uuid4().hex}.enc"
    write(name, data)
    return name


def write(name: str, data: bytes) -> None:
    path = _path(name)
    if path is None:
        raise ValueError("invalid file name")
    path.write_bytes(_fernet.encrypt(data))


def read(name: str) -> Optional[bytes]:
    path = _path(name) if name else None
    if path is None or not path.exists():
        return None
    raw = path.read_bytes()
    if not name.endswith(".enc"):
        return raw                                          # legacy unencrypted file
    try:
        return _fernet.decrypt(raw)
    except InvalidToken:
        return None


def derived_name(name: str, tag: str) -> str:
    """Name of a derived file (e.g. a heat-map) that is deleted together with `name`."""
    stem = name.rsplit(".", 1)[0]
    return f"{stem}.{tag}.enc"


def delete(name: Optional[str], *tags: str) -> None:
    if not name:
        return
    for n in [name] + [derived_name(name, t) for t in tags]:
        path = _path(n)
        if path is not None and path.exists():
            try:
                path.unlink()
            except OSError:
                pass
