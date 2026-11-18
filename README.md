# IntelliFlow — AI Data Intelligence

Public portfolio demo for customer-data quality checks, ML training workflows, and a React ops dashboard.

## What works on GitHub Pages

GitHub Pages hosts **static frontend only**. The public demo therefore:

- needs **no login**
- lets visitors **upload a CSV and inspect quality in the browser**
- keeps files on the visitor device (`localStorage`) — nothing is posted to a remote ML API from Pages
- enforces security limits: **CSV only · 2 MB · 5,000 rows · rate-limited checks**

Full model training / experiments / registry still need the FastAPI backend (local Docker or a separate host).

## Free public hosting (GitHub Pages)

GitHub Pages hosts the **frontend only** (no Python). CSV quality checks run in the browser on the same URL.

1. Push this repo to GitHub (public)
2. Repo **Settings → Pages → Build and deployment → Source: GitHub Actions**
3. Wait for the **Deploy GitHub Pages demo** workflow (Actions tab)
4. Open:

`https://<your-username>.github.io/<repo-name>/`

Example: `https://ersharadpatel.github.io/ai-data-intelligence/`

## One server (UI + API same URL)

For full training / predictions on **one port**:

```bash
# build UI then run API (serves frontend/dist + /api on :8001)
cd frontend && npm ci && npm run build && cd ..
uvicorn ai_data_intelligence.api:app --app-dir src --host 0.0.0.0 --port 8001
```

Or Docker (single container):

```bash
docker build -f Dockerfile.prod -t intelliflow .
docker run -p 8001:8001 intelliflow
```

Open `http://localhost:8001` — UI and API share the same origin (no separate Python URL).

## Local frontend demo

```bash
cd frontend
npm install
npm run dev
```

Open the Datasets page → **Try sample CSV** or upload your own CSV.

## Full stack (API + UI)

```bash
# API (example)
uvicorn ai_data_intelligence.api:app --app-dir src --port 8001

# UI
cd frontend
npm run dev
```

To force API-backed uploads instead of browser demo mode:

```bash
VITE_DEMO_MODE=false npm run dev
```

## Security controls (public demo)

| Control | Limit |
|---|---|
| Auth | None (intentional for demo) |
| File type | `.csv` only |
| Size | 2 MB |
| Shape | 5,000 rows / 40 columns |
| Rate limit | 8 checks / 10 minutes (per browser tab) |
| Storage | Browser local only on Pages |

Backend upload endpoint uses the same 2 MB / CSV / row-column limits when the API is online.
