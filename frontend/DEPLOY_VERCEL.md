# Deploying the frontend to Vercel

The React/Vite frontend deploys cleanly to Vercel. The FastAPI backend (`api/`)
does **not** — it needs a persistent, GPU-capable container (torch,
sentence-transformers, in-memory session state, 240 s analysis jobs), which
Vercel's serverless functions can't host. So this is a **split deploy**:

```
  Browser ──▶ Vercel (static React SPA)
                 │  /api/* calls
                 ▼
            Backend host (Hugging Face Spaces / Render / Railway)
            FastAPI + ML stack, reads the .h5 datasets
```

## 1. Deploy the backend first

The app is already configured for **Hugging Face Spaces** (see `README.md`
metadata) — that's the recommended host because it runs the full ML stack and
offers GPU tiers for the real embedding pipeline (`UAP_ANALYSIS_MODE=production`).
Render or Railway also work for CPU/mock mode.

Whichever you pick, set these env vars on the backend:

| Var | Value |
|-----|-------|
| `UAP_API_CORS_ORIGINS` | Your Vercel URL, e.g. `https://uap.vercel.app` (comma-separated for multiple) |
| `UAP_ANALYSIS_MODE` | `production` (real clustering) or `mock` (fast, no GPU) |

Note the backend's public URL — e.g. `https://<user>-uap.hf.space`.

## 2. Deploy the frontend to Vercel

1. **New Project** → import this repo.
2. Set **Root Directory** to `frontend` (the repo root is a Python project).
   Vercel auto-detects Vite from there; `vercel.json` pins the build.
3. Add an **Environment Variable**:

   | Name | Value |
   |------|-------|
   | `VITE_API_BASE` | The backend origin, e.g. `https://<user>-uap.hf.space` |

   `VITE_*` vars are inlined at build time, so changing it later needs a redeploy.
4. **Deploy.**

The client reads `VITE_API_BASE` (`src/api/client.ts`) and calls
`<VITE_API_BASE>/api/...`. When the var is unset it falls back to a same-origin
`/api`, so **local dev is unchanged** — `npm run dev` still proxies `/api` to
`localhost:8000` via `vite.config.ts`.

### Alternative: same-origin proxy (no CORS)

Instead of `VITE_API_BASE` + backend CORS, you can leave the env var unset and
proxy `/api` through Vercel. Add this to `vercel.json` **above** the SPA
fallback rewrite and you won't touch the backend's CORS at all:

```json
{ "source": "/api/:path*", "destination": "https://<user>-uap.hf.space/api/:path*" }
```

## CLI deploy (optional)

```bash
cd frontend
npx vercel            # preview deploy, prompts for login + project link
npx vercel --prod     # production deploy
```

## What does NOT go to Vercel

- `api/` (FastAPI) — backend host only.
- The `.h5` datasets and the 124 MB `uap_clusters_llm.html` — these stay with
  the backend, which serves the cluster viz via `/api/analysis/clusters`. They
  are not bundled into the 5 MB Vercel build.
