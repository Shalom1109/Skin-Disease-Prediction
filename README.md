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
- 7-class ensemble classifier with temperature-calibrated confidence and top-3 probabilities.
- Melanoma safety escalation with a **validated** threshold (fitted on the validation split by `evaluate.py`).
- Curated, schema-validated knowledge base (`data/knowledge_base.json`) + lexical retrieval for the chat assistant.
  An optional local LLM (Ollama) can phrase chat answers; it is **off** by default.
- Accounts (JWT, rate-limited login), private image storage, per-scan and full-account deletion.
- Professional, responsive UI (light/dark themes, mobile drawer navigation, accessible dialogs and toasts, printable report).
- Scan history, skin profile, ABCDE checklist, weather advice (Open-Meteo) and a clinic finder (OpenStreetMap). Location is only
  requested when you press a button.
- Reproducible research pipeline: lesion-grouped splits, balanced-accuracy model selection, full evaluation report.

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
45 tests cover auth, rate limiting, upload validation, privacy (image ownership, EXIF stripping, deletion), the
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
- `python sql_console.py` - read-only database browser (`--write` to modify, with confirmation).
- `python scripts/check_model_forward.py` - builds the ensemble and runs one forward pass.

## License
MIT
