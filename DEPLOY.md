# Deploying UFOSINT (React + FastAPI)

The two halves deploy to different hosts because they have different needs:

| Part | Host | Why |
|------|------|-----|
| React frontend (`frontend/`) | **Vercel** | Static Vite SPA — Vercel's sweet spot. |
| FastAPI/uvicorn backend (`api/main.py`) | **Render or Railway** | Long-lived process, heavy deps (torch, sentence-transformers, UMAP), a 1.8 GB dataset, multi-minute requests. None of this fits Vercel's serverless model. |

```
Vercel (static React)  ──VITE_API_BASE──▶  Render / Railway (uvicorn, Dockerfile.api)
```

Render and Railway are interchangeable here — both build `Dockerfile.api`, run a
long-lived container, inject `$PORT`, and offer a persistent disk/volume for the
dataset. Pick one (§1 Render, §1b Railway); the config for both lives in the repo.

Deploy the **backend first** so you have its URL when configuring the frontend.

---

## 1. Backend → Render

Files: `render.yaml` (blueprint), `render-start.sh` (entrypoint), `Dockerfile.api` (image).

1. Push this repo to GitHub.
2. Render dashboard → **New → Blueprint** → select the repo. Render reads `render.yaml` and creates the `ufosint-api` web service with a 10 GB disk at `/data`.
3. First build takes a while (CPU torch + ML wheels). When it's up, note the URL, e.g. `https://ufosint-api.onrender.com`.
4. Test: `curl https://ufosint-api.onrender.com/api/dashboard/summary` should return JSON.

### What needs no configuration
- **API keys** (OpenAI / Cohere / Gemini) are entered in the **UI per request** and never stored on the server — no secret env vars required.
- Parsing, RAG search, and SCU normalization work immediately (they act on uploaded data + your keys).

### The dataset (the one real decision)
The pre-parsed `parsed_files_distance_embeds.h5` is **1.8 GB and git-ignored**, so it isn't in the image. The **Data Explorer "load dataset"** action and cluster analysis need it; everything else doesn't. Options:

- **Skip it for now** — deploy as-is. The Data Explorer returns 404 until a dataset is provided. Good enough to demo parsing/RAG/SCU.
- **Host the file and auto-download** — put the `.h5` somewhere with a direct/presigned URL (Cloudflare R2, S3, Backblaze B2, or a Hugging Face dataset repo), then set on the Render service:
  - `DATASET_URL_WEST` → URL to `parsed_files_distance_embeds.h5` (1.8 GB)
  - `DATASET_URL_EAST` → URL to `final_ufoseti_dataset.h5` (38 MB)

  `render-start.sh` downloads them to the `/data` disk on first boot only (persists across restarts). Redeploy after setting them.
- **Upload directly to the disk** — use Render's shell to `curl`/`scp` the files into `/data` once.

### Plan / RAM
`render.yaml` defaults to **`standard` (2 GB)**, which boots and runs the API-key features fine. The full embedding + UMAP + HDBSCAN pipeline over the 1.8 GB dataset wants **`pro` (4 GB)** or **`pro plus` (8 GB)** — bump `plan:` in `render.yaml` or in the dashboard. The free tier is **not** enough (it sleeps and lacks RAM/disk).

---

## 1b. Backend → Railway (alternative to Render — pick one)

Files: `railway.json` (build + start config), `render-start.sh` (entrypoint, shared with Render), `Dockerfile.api` (image).

Railway is the same container model as Render. The difference: `railway.json` only
covers build/start/healthcheck — the **volume and env vars are set in the dashboard
(or `railway` CLI)**, not in a blueprint.

1. **railway.com → New Project → Deploy from GitHub repo** → pick `UAP_Analysis_Tool`, branch `deploy/render-vercel`. Railway reads `railway.json` and builds `Dockerfile.api`.
2. **Add a Volume** (service → *Settings → Volumes* or *+ Volume*) mounted at **`/data`** so the dataset + model cache persist across deploys.
3. **Set environment variables** (service → *Variables*) — same set as Render, minus the plan/disk (the volume covers that):
   - `UAP_ANALYSIS_MODE=production`
   - `HF_HOME=/data/huggingface`
   - `UAP_DATASET_WEST=/data/parsed_files_distance_embeds.h5`
   - `UAP_DATASET_EAST=/data/final_ufoseti_dataset.h5`
   - `UAP_API_CORS_ORIGINS=` your Vercel URL(s) (set after the frontend exists — see §3)
   - *(optional)* `DATASET_URL_WEST` / `DATASET_URL_EAST` — direct/presigned `.h5` URLs; `render-start.sh` fetches them to `/data` on first boot. Leave unset to deploy without the dataset (parsing/RAG/SCU still work).
4. **Generate a domain** (service → *Settings → Networking → Generate Domain*) and note it, e.g. `https://ufosint-api.up.railway.app`.
5. Test: `curl https://<your-domain>/api/dashboard/summary` should return JSON.

Notes:
- Railway injects `$PORT`; `render-start.sh` already binds `uvicorn` to it (`render-start.sh` is just the historical name — it's platform-neutral).
- The same dataset / RAM guidance from §1 applies. Railway bills by usage; give the service enough memory (≥4 GB) before running the full embedding pipeline.
- `railway.json` pins the Dockerfile so Railway doesn't try to auto-detect (Nixpacks) and miss `Dockerfile.api`.

---

## 2. Frontend → Vercel

`frontend/vercel.json` is already set up (Vite framework + SPA rewrites).

1. Vercel → **Add New → Project** → import the repo.
2. **Root Directory: `frontend`** (important — the app is in a subfolder).
3. Add an environment variable:
   - `VITE_API_BASE` = your Render URL, e.g. `https://ufosint-api.onrender.com` (no trailing slash).
   This is inlined at build time by `src/api/client.ts`, so the frontend calls the Render backend directly.
4. Deploy. Note the resulting URL, e.g. `https://ufosint.vercel.app`.

---

## 3. Wire CORS (close the loop)

The backend only accepts browser requests from origins in its allowlist. After the frontend URL exists:

1. On the backend service (Render `ufosint-api` **or** the Railway service), set `UAP_API_CORS_ORIGINS` to your Vercel URL(s), comma-separated — include both the production domain and any custom domain, e.g.
   `https://ufosint.vercel.app,https://ufosint.yourdomain.com`
2. Redeploy the backend (both Render and Railway auto-deploy on an env change).

Done. Visiting the Vercel URL serves the React app, which talks to the backend.

---

## Notes & gotchas
- **Cold starts:** the first request after idle re-loads torch and (lazily) the e5-large model. With the disk-backed `HF_HOME`, the model is cached after the first download.
- **Long requests:** UMAP/HDBSCAN and the magnetic BGS fetches can run for minutes. Render and Railway allow this; Vercel would not.
- **Re-deploys** don't wipe `/data` (Render disk / Railway volume), so datasets and the model cache survive.
- **Local dev is unchanged:** without the env overrides the paths fall back to the repo-root `.h5` files, and `docker compose up` still works as before.
