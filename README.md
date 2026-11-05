# SkinSense AI - Skin Lesion Screening Demo

SkinSense AI classifies a skin-lesion photo into one of seven HAM10000 categories with a triple-backbone ensemble
(**ConvNeXt-B + ViT-B + MaxViT-B**), then shows a curated clinical summary, keeps a private scan history and offers a
nearby-dermatologist finder.

> **Not a medical device.** It is an educational screening aid trained on dermoscopic images. It cannot diagnose
> anything. See [MODEL_CARD.md](MODEL_CARD.md) for intended use and limitations.
> Measured on a held-out, lesion-grouped test split of HAM10000: **80.2 % accuracy, 72.7 % balanced accuracy**,
> melanoma AUC 0.92, and 90.8 % melanoma sensitivity at the safety threshold (with about 27 % false alarms).
> That is a demo-level result on dermoscopy images, not clinical performance. Details and caveats: [MODEL_CARD.md](MODEL_CARD.md).

## Features
**For patients**
- Upload a photo and get a plain-language **referral** (see a dermatologist promptly / book a review / likely harmless,
  re-check in 3 months / retake the photo) together with the 7-class probabilities. The melanoma safety rule uses a
  threshold validated on held-out data.
- **Instant photo-quality hints** (blur, darkness, glare, resolution) before you spend an analysis on a bad photo.
- **Model focus heat-maps** (Grad-CAM) that show which part of the photo influenced the result.
- **Track a spot over time**: group scans into a lesion, compare first vs latest photo, and get a flag when the
  melanoma probability rises (it does not measure growth).
- **Care team**: share your scans with a clinician you choose, revoke at any time, and read the **access log** of
  everyone who viewed your data. Delete single scans or your whole account whenever you like.
- Optional context (age, sex, body site), skin profile, ABCDE checklist, weather advice and a clinic finder.
  Location is only requested when you press a button.

**For clinicians** (accounts are promoted by an administrator, see `docs/DEPLOYMENT.md`)
- A workspace of the patients who shared their scans: photos, model output, heat-maps, and a review form
  (confirm or correct the label, add a note). Every access is written to the patient's audit log.

**Under the hood**
- Triple-backbone ensemble (ConvNeXt-B, ViT-B, MaxViT-B), temperature-calibrated, running on the GPU when available.
- Accounts (JWT, rate-limited login), photos **encrypted at rest**, strict per-user access, SQLite or PostgreSQL.
- Curated, schema-validated knowledge base and a chat assistant (optional local LLM, off by default).
- Responsive UI with light/dark themes, accessible dialogs and toasts, printable report.
- Reproducible research pipeline: lesion-grouped splits, balanced-accuracy model selection, evaluation with
  95% confidence intervals, abstention analysis, and tools to compare other backbones fairly
  (`scripts/extract_embeddings.py`, `scripts/train_embedding_head.py`, `scripts/train_fusion.py`).

## Setup
Python 3.10+.

```bash
pip install -r requirements.txt          # run the app
pip install -r requirements-train.txt    # + training / evaluation
pip install -r requirements-dev.txt      # + tests
```

**Weights** (not in git): place `skin_sense_ensemble_best.pth` in `backend/` (or set `SKINSENSE_WEIGHTS=/path/file.pth`).
Without weights the API starts but `/predict` answers **503** - it never silently predicts with random weights
(`SKINSENSE_ALLOW_UNTRAINED=1` overrides this for development only).

## Run
Windows: double-click `LAUNCH_SKINSENSE_AI.bat` (or `Open_SkinSense.vbs` for a hidden background server), or:

```bash
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Open <http://127.0.0.1:8000>. `Stop_SkinSense.bat` stops the server.

## Train and evaluate properly
1. Put HAM10000 in an `ImageFolder` layout (`dataset/akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`) and download `HAM10000_metadata.csv`.
2. Train: `python train.py --data-dir dataset --metadata HAM10000_metadata.csv [--finetune]`
3. Evaluate + calibrate: `python evaluate.py --data-dir dataset --metadata HAM10000_metadata.csv`
   -> `reports/evaluation.md`, `reports/confusion_matrix.png`, and `backend/calibration.json` (loaded by the API).
4. Optional cross-validation: `python train_kfold.py --data-dir dataset --metadata HAM10000_metadata.csv`

The metadata CSV matters: HAM10000 has several photos per lesion, and splitting by image leaks the same lesion into
train and test. With the CSV, splits are grouped by `lesion_id` and checked for overlap.

## Configuration (environment variables)
| Variable | Default | Purpose |
|---|---|---|
| `SKINSENSE_SECRET_KEY` | random, stored in `backend/.secret_key` | JWT signing key |
| `SKINSENSE_ORIGINS` | `http://127.0.0.1:8000,http://localhost:8000` | allowed CORS origins |
| `SKINSENSE_WEIGHTS` | auto (`*best*`, then `*final*` in `backend/`) | model file |
| `SKINSENSE_MAX_UPLOAD_MB` | 8 | upload limit |
| `SKINSENSE_DEVICE` | `auto` | `auto` uses the GPU when CUDA is available (about 0.5 s per image vs 5 s on CPU); `cpu` forces CPU |
| `SKINSENSE_TOKEN_MINUTES` | 60 | login lifetime |
| `SKINSENSE_LLM` / `SKINSENSE_LLM_URL` / `SKINSENSE_LLM_MODEL` | off / - / `llama3` | optional LLM for chat (`SKINSENSE_LLM=ollama`; needs `pip install openai`) |
| `SKINSENSE_DB`, `SKINSENSE_UPLOADS` | `backend/skinsense.db`, `backend/uploads` | storage locations |

## Tests
```bash
python -m pytest -q
```
90 tests cover auth, rate limiting, upload validation, privacy (image ownership, EXIF stripping, deletion), the
decision logic, calibration metrics, lesion-grouped splits, the knowledge base and front-end consistency (every id the
script uses exists, icons are defined, assets exist, valid JS). They use a fake model, so no
weights or network are needed. `scripts/smoke_predict.py` is a manual end-to-end check against a running server
(it is *not* an accuracy test).

## Architecture
```
Browser (HTML/CSS/JS) --JWT--> FastAPI (backend/app.py)
                                  |-- inference.py  preprocess, temperature, escalation rule  <- calibration.json
                                  |-- model.py      ConvNeXt + ViT + MaxViT ensemble
                                  |-- rag.py        curated KB + retrieval (+ optional local LLM for chat)
                                  |-- database.py   SQLite: users, history
                                  `-- security.py   rate limiter
Offline: train.py / train_kfold.py / evaluate.py share backend/data_utils.py (grouped splits) and backend/metrics.py
```

## Utilities
- `python scripts/manage_users.py make-clinician <username>` - promote an account (clinicians are never self-service).
- `python scripts/calibrate_quality.py` - check the photo-quality thresholds against your own images.
- `python scripts/train_fusion.py` - test whether age/sex/body-site context improves the model (exports only if you ask).
- `python sql_console.py` - read-only database browser (`--write` to modify, with confirmation).
- `python scripts/check_model_forward.py` - builds the ensemble and runs one forward pass.

## Deployment
Docker + PostgreSQL + HTTPS files and a checklist of what is verified are in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## License
MIT
