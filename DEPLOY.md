# Deploying UFOSINT (React + FastAPI)

The two halves deploy to different hosts because they have different needs:

| Part | Host | Why |
|------|------|-----|
| React frontend (`frontend/`) | **Vercel** | Static Vite SPA — Vercel's sweet spot. |
| FastAPI/uvicorn backend (`api/main.py`) | **Render** | Long-lived process, heavy deps (torch, sentence-transformers, UMAP), a 1.8 GB dataset, multi-minute requests. None of this fits Vercel's serverless model. |

```
Vercel (static React)  ──VITE_API_BASE──▶  Render (uvicorn, Dockerfile.api)
```

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

1. On the Render `ufosint-api` service, set `UAP_API_CORS_ORIGINS` to your Vercel URL(s), comma-separated — include both the production domain and any custom domain, e.g.
   `https://ufosint.vercel.app,https://ufosint.yourdomain.com`
2. Redeploy the backend (or it auto-deploys on the env change).

Done. Visiting the Vercel URL serves the React app, which talks to the Render backend.

---

## Notes & gotchas
- **Cold starts:** the first request after idle re-loads torch and (lazily) the e5-large model. With the disk-backed `HF_HOME`, the model is cached after the first download.
- **Long requests:** UMAP/HDBSCAN and the magnetic BGS fetches can run for minutes. Render allows this; Vercel would not.
- **Re-deploys** don't wipe `/data`, so datasets and the model cache survive.
- **Local dev is unchanged:** without the env overrides the paths fall back to the repo-root `.h5` files, and `docker compose up` still works as before.
