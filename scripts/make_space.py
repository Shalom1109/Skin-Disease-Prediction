"""Build a ready-to-push Hugging Face Space (Docker) from this repository.

    python scripts/make_space.py --out dist/space

The bundle contains only what the running app needs: the backend code, the validated calibration files, the front end,
the knowledge base, a Dockerfile and a README with the Space metadata. It deliberately does NOT contain model weights,
datasets, databases, keys or tests. The weights are downloaded at start-up from a private Hugging Face model repo
(SKINSENSE_WEIGHTS_REPO), see docs/FREE_DEPLOYMENT.md.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

README = """---
title: SkinSense AI
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: other
short_description: Educational skin-lesion screening demo. Not a medical device.
---

# SkinSense AI

An educational demonstration of skin-lesion image screening with a three-model ensemble
(ConvNeXt-B, ViT-B, MaxViT-B). **It is not a medical device and cannot diagnose anything.**
Do not upload photos you are not comfortable sharing with a third-party host, and never rely on a
result instead of seeing a doctor.

* Trained on HAM10000 (dermoscopy images; licensed for non-commercial use).
* Free CPU hosting: the first request after a sleep takes about a minute and an analysis takes tens of seconds.
* Accounts and photos on this free tier are not guaranteed to survive a restart.

Source code, evaluation and limitations: https://github.com/Shalom1109/Skin-Disease-Prediction
"""

BACKEND_FILES = ["calibration.json", "fusion.json"]


def build(out: Path):
    if out.exists():
        shutil.rmtree(out)
    (out / "backend").mkdir(parents=True)

    for py in sorted((ROOT / "backend").glob("*.py")):
        shutil.copy2(py, out / "backend" / py.name)
    for name in BACKEND_FILES:
        src = ROOT / "backend" / name
        if src.exists():
            shutil.copy2(src, out / "backend" / name)

    shutil.copytree(ROOT / "frontend", out / "frontend")
    (out / "data").mkdir()
    shutil.copy2(ROOT / "data" / "knowledge_base.json", out / "data" / "knowledge_base.json")
    for name in ("requirements.txt", "requirements-deploy.txt", ".dockerignore"):
        shutil.copy2(ROOT / name, out / name)

    # The Space page shows the app inside an iframe, so allow huggingface.co to embed it (and nobody else).
    docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    marker = "ENV SKINSENSE_UPLOADS=/data/uploads"
    if docker.count(marker) != 1:
        sys.exit("Dockerfile no longer contains the expected ENV line; update scripts/make_space.py")
    # `ENV a=1 b=2` (several pairs on one instruction) is valid Dockerfile syntax
    docker = docker.replace(marker, "ENV SKINSENSE_FRAME_ANCESTORS=https://huggingface.co SKINSENSE_UPLOADS=/data/uploads", 1)
    (out / "Dockerfile").write_text(docker, encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dist/space")
    args = ap.parse_args()
    out = build(Path(args.out))
    files = [p for p in out.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files) / 1024 / 1024
    print(f"Built {out}: {len(files)} files, {size:.1f} MB")
    print("Next: see docs/FREE_DEPLOYMENT.md (create the Space, upload the weights, push this folder).")


if __name__ == "__main__":
    main()
