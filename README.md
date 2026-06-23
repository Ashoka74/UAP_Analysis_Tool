---
title: UFOSINT
emoji: 🛸
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: 1.36.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: UFO/UAP AI Analyst
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

---

## Modern FrontEnd (React + FastAPI)

The repo ships a decoupled web stack alongside the Streamlit app:

- **Backend** — FastAPI service under `./api` (routes in `api/routes/`, services in `api/services/`, models in `api/models/`).
- **Frontend** — Vite + React + TypeScript + Tailwind + Zustand under `./frontend`, talking to the backend through a Vite proxy.

### Prerequisites
- Python 3.11+ with `fastapi` and `uvicorn` installed (covered by `requirements.txt` / `pyproject.toml`).
- Node 20+ for the frontend.

### 1. Start the backend (FastAPI)

From the repo root:

```bash
uvicorn api.main:app --reload --port 8000
```

The `--reload` flag picks up code changes automatically.

### 2. Start the frontend (Vite + React)

```bash
cd frontend
npm install      # first time only
npm run dev
```

Vite starts on port **5173** and proxies every `/api/*` request to `http://localhost:8000` (see `frontend/vite.config.ts`), so the React app calls FastAPI transparently — no CORS configuration needed.

### 3. Open the app

http://localhost:5173

### Production build

```bash
cd frontend
npm run build       # output in frontend/dist
```

Serve the `dist/` artifacts behind any static host and keep `uvicorn api.main:app` running for the API.