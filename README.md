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

Anyone can open the site and check a CSV — no paid hosting needed.

1. Push this repo to GitHub (public)
2. Repo **Settings → Pages → Build and deployment → Source: GitHub Actions**
3. Wait for the **Deploy GitHub Pages demo** workflow (Actions tab)
4. Open:

`https://<your-username>.github.io/<repo-name>/`

Example if the repo is `ai-data-intelligence`:

`https://<your-username>.github.io/ai-data-intelligence/`

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
