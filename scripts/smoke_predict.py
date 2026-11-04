"""Smoke test for a RUNNING server (python -m uvicorn app:app, from backend/).

Uploads the seven demo illustration images from frontend/img/diseases and prints what the
API returns. This only checks that the pipeline works end to end.

!! It is NOT an accuracy test. These pictures are demo illustrations, not part
!! of a held-out test set. For real numbers use:  python evaluate.py

    python scripts/smoke_predict.py --user demo_user --password "Demo@1234"

The account is created automatically if it does not exist.
"""
import argparse
from pathlib import Path

import requests

IMG_DIR = Path(__file__).resolve().parent.parent / "frontend" / "img" / "diseases"
SAMPLES = {
    "Actinic Keratoses": "Actinic Keratoses.jpg",
    "Basal Cell Carcinoma": "Basal Cell Carcinoma.jpg",
    "Benign Keratosis": "Benign Keratosis.jpg",
    "Dermatofibroma": "Dermatofibroma.jpg",
    "Melanoma": "Melanoma.jpg",
    "Nevi": "Nevi.jpg",
    "Vascular": "Vascular Lesions.jpg",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--user", default="demo_user")
    ap.add_argument("--password", required=True)
    args = ap.parse_args()

    requests.post(f"{args.url}/register", data={"username": args.user, "password": args.password})
    r = requests.post(f"{args.url}/login", data={"username": args.user, "password": args.password})
    r.raise_for_status()
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

    print(f"{'Illustration':<24} | {'Prediction':<38} | Confidence")
    print("-" * 80)
    for label, filename in SAMPLES.items():
        path = IMG_DIR / filename
        if not path.exists():
            print(f"{label:<24} | (missing file {path})")
            continue
        with open(path, "rb") as f:
            resp = requests.post(f"{args.url}/predict", headers=headers, files={"file": (filename, f, "image/jpeg")})
        if resp.ok:
            d = resp.json()
            print(f"{label:<24} | {d['prediction']:<38} | {d['confidence']}%")
        else:
            print(f"{label:<24} | ERROR {resp.status_code}: {resp.text[:80]}")


if __name__ == "__main__":
    main()
