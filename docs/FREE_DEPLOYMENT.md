# Free deployment (Hugging Face Space)

The cheapest way to get the **whole app** online for nothing is a **Hugging Face Space** (Docker, free "CPU basic":
2 vCPU, 16 GB RAM, no credit card). It serves the API *and* the web page from one address, so no Vercel is needed
(Vercel is optional, see the end).

> **What I verified and what I could not.** I built the Space bundle, ran the app *from the bundle* on CPU only
> (page, login, analysis, headers all work) and tested the weights download with a stand-in. I could **not** run the
> Hugging Face steps themselves: there is no Hugging Face account on the development machine. The commands below follow
> Hugging Face's CLI; I checked the option names against the `hf` command installed on this machine (version 1.5). Older
> versions call it `huggingface-cli` instead of `hf`. If one differs, the website offers the same actions in the browser.

## What to expect from the free tier (read this first)
| | Reality |
|---|---|
| Speed | About 5 s per analysis on a 12-core laptop CPU; a 2-vCPU free Space is probably **15-30 s** (my estimate, not measured) |
| Cold start | The Space sleeps after about 2 days without visitors. The next visit takes **1-2 minutes** (weights download + model load) |
| Heat-maps | **Off** on CPU hosts (one took 25 s even on a 12-core laptop). The button is hidden automatically |
| Storage | The disk is **wiped on restart or rebuild**: accounts, scans and photos disappear. A free external PostgreSQL (e.g. Neon, Supabase) keeps accounts via `SKINSENSE_DATABASE_URL`, but photos are still lost |
| Privacy | A third party hosts everything you upload. **Never use real patient photos or data.** Demo images only |
| Licence | HAM10000 (the training data) is, to my knowledge, licensed for **non-commercial** use. Check it before any commercial use |

## Step 1: Accounts and tokens (you do this)
1. Create a free account at huggingface.co.
2. Settings -> Access Tokens -> create a **Write** token (to upload) and a separate **Read** token (the Space will use it
   to fetch your private weights). Treat both like passwords and never commit them.
3. On your computer: `pip install -U huggingface_hub`, then `hf auth login` and paste the Write token
   (older versions: `huggingface-cli login`).

## Step 2: Upload the model weights to a *private* model repo
The 1.2 GB weights cannot go on GitHub (100 MB limit) and should not be public.
```bash
hf upload YOUR_USERNAME/skinsense-weights backend/skin_sense_ensemble_best.pth skin_sense_ensemble_best.pth --private
```
This creates the private repo `YOUR_USERNAME/skinsense-weights` and uploads the file (about 1.2 GB, so it takes a while).

## Step 3: Create the Space
huggingface.co -> New Space -> name `skinsense`, SDK **Docker** (template "Blank"), hardware
**CPU basic - 2 vCPU - 16 GB - FREE**, visibility Public.

## Step 4: Build the bundle and upload it
```bash
python scripts/make_space.py --out dist/space
hf upload YOUR_USERNAME/skinsense dist/space . --repo-type space
```
`dist/space` is 5 MB: backend code, the validated calibration files, the web page and a Dockerfile. It contains **no**
weights, keys, datasets or databases.

## Step 5: Set the secrets and variables
Space -> Settings -> Variables and secrets.

| Kind | Name | Value |
|---|---|---|
| Secret | `SKINSENSE_SECRET_KEY` | `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| Secret | `SKINSENSE_IMAGE_KEY` | `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| Secret | `HF_TOKEN` | your **Read** token |
| Variable | `SKINSENSE_WEIGHTS_REPO` | `YOUR_USERNAME/skinsense-weights` |

Setting the two keys as secrets matters: without them the app invents new ones on every restart and every login becomes
invalid. Optional: `SKINSENSE_DATABASE_URL` (free external PostgreSQL) so accounts survive restarts.

## Step 6: Wait and open it
The first build takes about 10 minutes (installing PyTorch). Then the app downloads the weights and loads the model
(1-2 minutes). Your app is at `https://YOUR_USERNAME-skinsense.hf.space`. `/health` should say `"model_loaded": true`.

If something fails, open the Space's **Logs** tab. Common causes: a wrong repo name in `SKINSENSE_WEIGHTS_REPO`, a Read
token without access to the private repo, or running out of memory (not expected: the app uses about 3 GB).

## Optional: put the front end on Vercel
Not needed (the Space already serves the page), but if you want the Vercel address: follow "Vercel (front end) + a
separate API host" in `DEPLOYMENT.md` with `SKINSENSE_API_URL=https://YOUR_USERNAME-skinsense.hf.space` on Vercel and
`SKINSENSE_ORIGINS=https://<your-project>.vercel.app` as a Space variable.

## Other free options
* **Oracle Cloud "Always Free" Ampere VM** (up to 4 cores, 24 GB RAM): always on, with a persistent disk, so accounts and
  photos survive. It needs a card for sign-up and some server administration (use `docker-compose.yml`). Not tested here.
* Render, Railway and Fly.io free tiers have far too little memory for this model (the app needs about 3 GB).
