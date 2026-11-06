# Deployment guide

What is verified and what is not, so you know where to look first if something misbehaves.

| Piece | Status |
|---|---|
| App on SQLite (local, `LAUNCH_SKINSENSE_AI.bat`) | Verified, used daily in development |
| App on **PostgreSQL** | **Verified**: the full 84-test suite passes against PostgreSQL 18 |
| `Dockerfile`, `docker-compose.yml`, `Caddyfile` | **Written but never run** (Docker is not installed on the development machine). Expect to fix small things on first build. |
| GPU inference | Verified on an RTX 4050 (about 0.5 s per image, vs 5 s on CPU) |
| Front end on Vercel + API elsewhere (CORS, config, offline banner) | **Verified locally** with page and API on different ports; the actual Vercel deployment depends on your account |

## 1. Secrets (do this first)
| Variable | What it protects | How to create |
|---|---|---|
| `SKINSENSE_SECRET_KEY` | login tokens | `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `SKINSENSE_IMAGE_KEY` | stored photos (encrypted at rest) | `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `POSTGRES_PASSWORD` | database | any long random string |

**Back the image key up somewhere separate from the server.** If it is lost, every stored photo is permanently
unreadable. Rotating it also makes old photos unreadable, so re-encrypt first if you ever need to change it.

## 2. Run with Docker (PostgreSQL + the app)
```bash
cp .env.example .env            # fill in the secrets above and SKINSENSE_ORIGINS (your public URL)
mkdir models                    # put skin_sense_ensemble_best.pth here; calibration.json is already in the image
docker compose up -d --build
```
Open `http://localhost:8000`. For automatic HTTPS on a public domain set `DOMAIN` in `.env` and run
`docker compose --profile https up -d`. Do not expose the app over plain HTTP on the internet: logins and
medical photos would travel unencrypted.

GPU host: build with `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124` and run the container with the
NVIDIA container runtime. Without a GPU the app works but takes about 5 s per image.

## Vercel (front end) + a separate API host
**Vercel cannot run this project's backend.** The API needs PyTorch (over 2 GB installed) and a 1.2 GB model file,
while a Vercel Python function is limited to about 250 MB, has no persistent disk and short time limits. What Vercel
*can* do is host the static front end (`frontend/`), which then talks to the API running somewhere else.

| Piece | Where it runs | Status |
|---|---|---|
| `frontend/` (HTML, CSS, JS) | **Vercel** (free tier is fine) | Verified locally in the same split topology (page and API on different origins) |
| API + model + database + encrypted photos | A container host with 4 GB+ RAM (Render, Railway, Fly.io, a VPS, Hugging Face Spaces Docker), or your own machine behind a tunnel for demos | `Dockerfile` written, not run (see above) |

### Steps
1. **Run the API somewhere** (section 2 above) and note its public **https** URL, e.g. `https://skinsense-api.example.com`.
   Plain `http://` will not work from an https Vercel site (mixed content).
2. **Import the repository in Vercel**: vercel.com/new -> import `Skin-Disease-Prediction`. `vercel.json` already sets
   the build command and the output folder (`frontend`); leave the framework preset as "Other".
3. **Add one environment variable** in the Vercel project: `SKINSENSE_API_URL` = the API URL from step 1. The build step
   (`scripts/vercel_build.mjs`) writes it into `frontend/js/config.js`. Redeploy after changing it.
4. **Allow the site on the API**: set `SKINSENSE_ORIGINS=https://<your-project>.vercel.app` (comma-separate several
   origins) on the API host and restart it. Without this, browsers block every request (CORS).
5. Open the Vercel URL. If the API is unreachable the page shows a banner saying so, instead of failing silently.

Where the API gets its weights is up to the host: mount them as a volume (`/models`, as in `docker-compose.yml`) or bake
them into a private image. They must never go into the public Git repository (GitHub's file limit is 100 MB, and the
model is trained on a licensed dataset).

### Demo from your own computer
Run the API locally (`LAUNCH_SKINSENSE_AI.bat`), expose it with a tunnel such as `cloudflared tunnel --url
http://127.0.0.1:8000`, and use the tunnel's https URL as `SKINSENSE_API_URL`. This works only while your computer is on
and awake, and everything (including patient photos) passes through the tunnel provider, so it is for demos, not for
real patients.

## 3. Run without Docker on PostgreSQL
```bash
pip install -r requirements-deploy.txt
export SKINSENSE_DATABASE_URL="postgresql+psycopg2://user:password@host:5432/skinsense"
cd backend && python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Tables are created automatically, and older databases are migrated in place (new columns are added on start-up).

## 4. Create clinician accounts
Anyone can register as a patient. Clinicians are promoted by an administrator **after verifying who they are**:
```bash
python scripts/manage_users.py make-clinician dr_username
```
Patients then share their scans with that username from the "Care team" card. Every clinician view is recorded in
the patient's access log.

## 5. Backups
Back up together: the database, the `uploads` volume (encrypted photos) and the image key (separately, securely).
A database backup without the photos or the key cannot be fully restored.

## 6. Known limits before real patients use it
* **Single worker only.** The login rate limiter and the model live in one process's memory. Do not run several
  `uvicorn --workers`; scale with a queue or a gateway if you outgrow one process.
* Login tokens are kept in the browser's `localStorage` (mitigated by escaping all rendered text); httpOnly cookies
  with CSRF protection, two-factor sign-in and token refresh are not implemented.
* No email verification or password reset.
* The model is a demo-level classifier (see `MODEL_CARD.md`). It has not been validated on phone photos or across
  skin tones, and no medical-device clearance exists. Do not market it as diagnostic.
* Privacy law (consent wording, retention period, data-protection officer, cross-border storage) depends on your
  country and must be reviewed by someone qualified before collecting real patient data.
