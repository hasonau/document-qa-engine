# Deploy later (not tested on a real host)

This project is meant to run locally for review. Backend hosting was deferred after a memory audit.

**Every step below is marked: not tested on a real host.**

## Memory audit (why not a 512 MB free tier)

Measured locally on Windows, Python 3.14.2, CPU-only torch, spare port without `--reload`. Working set from `Get-Process` `WorkingSet64` on the uvicorn worker process (the parent launcher process was only a few MB).

| Stage | Working set |
|--------|-------------|
| After startup + `GET /healthz` | about **490 MB** |
| After one small PDF upload (1 chunk, no Groq call) | about **560 MB** |

**What that means:** plan on a host with **at least 1–2 GB RAM** so idle load, ingest, and query (rerank) have headroom. A **512 MB** free tier is a poor fit:

1. **Memory limit** — idle alone is already near 512 MB; one upload exceeded it in the local measurement.
2. **Spin-down when idle** — free web services often sleep; the first request after sleep can take a minute while models load.
3. **Non-durable disk** — free disks are usually ephemeral. Local `chroma_db/`, `bm25_*.pkl`, and `month2_api_service/documents/` can disappear on restart. Accept that for demos, or move later to durable object/vector storage.

## Backend (when you do host it)

**Not tested on a real host.**

### Service shape

- Python web service, start from the **repo root** (paths for `chroma_db/`, `bm25_*.pkl`, and `month2_api_service/documents/` are relative to the working directory).
- Health check: `GET /healthz` → `{"status":"ok"}`.
- Bind `0.0.0.0` and read `PORT` (already supported in `month2_api_service/main.py`).

### Python version

Local `.venv` used **3.14.2**. Many hosts do not offer 3.14 yet. Pin a version **the host actually lists** (for example 3.12.x) in whatever file that host uses (`runtime.txt`, `.python-version`, or the dashboard). Re-test install and startup after changing the pin. **Not tested on a real host.**

### Install (CPU-only torch)

`sentence-transformers` pulls `torch`. Prefer the **CPU** wheel so the image stays smaller:

```text
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Exact index URL and pin should match the Python version on the host. **Not tested on a real host.**

### Pre-download models at build time

Embedding model: `all-MiniLM-L6-v2`. Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (see `month2_api_service/services/rag.py`).

Idea for a future build step (not wired in this lite pass): set `HF_HOME` to a folder inside the project, run a short script that loads `SentenceTransformer` and `CrossEncoder` once during build, then start the app with `HF_HUB_OFFLINE=1` so a cold start does not download weights. **Not tested on a real host.**

### Build and start commands (examples)

Build (dependency install + optional model warm):

```text
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
# optional: python scripts/download_models.py
```

Start from repo root:

```text
python -m uvicorn month2_api_service.main:app --host 0.0.0.0 --port $PORT
```

On Windows locally, `$PORT` is `%PORT%` or omit and use the default 8000. **Not tested on a real host.**

### Environment variables (placeholders only)

| Variable | Example placeholder | Notes |
|----------|---------------------|--------|
| `GROQ_API_KEY` | `your_groq_api_key_here` | Required for live answers |
| `ALLOWED_ORIGINS` | `https://your-frontend.example` | Comma-separated; must match the browser origin |
| `PORT` | `8000` | Provided by many hosts automatically |

Optional limits already documented in `.env.example` (`MAX_UPLOAD_BYTES`, rate limits, etc.) stay available. **Not tested on a real host.**

## Frontend + cookies across domains

**Not tested on a real host.**

The UI uses a session cookie (`session_id`) with `credentials: "include"`. If the frontend and API are on **different sites**, browsers may not send that cookie as a first-party cookie.

**Preferred fix:** put the API behind the same origin as the frontend (for example a rewrite/proxy: browser calls `/api/*` on the frontend host; the host forwards to the Python service). Then the cookie stays first-party.

**Fallback:** `SameSite=None; Secure` on the cookie plus `ALLOWED_ORIGINS` set to the **exact** frontend origin (and CORS credentials enabled — already true in the app). Some browsers still block or partition third-party cookies; the proxy approach is more reliable.

## Smoke-test checklist (after a future deploy)

**Not tested on a real host.**

1. `GET /healthz` returns ok.
2. `POST /load-sample` (or upload a PDF) succeeds; knowledge base lists the document.
3. Ask one of the suggested questions; an answer streams or returns with sources.
4. Delete the document; it disappears from the list.
5. Trigger a rate-limit message (many uploads/queries) and confirm a clear error.
6. Refresh the browser; the session cookie still lists the same documents (if disk was not wiped).
7. After a host restart, confirm whether documents survived (expect loss on ephemeral disk).

## Troubleshooting (expected later)

| Symptom | Likely cause |
|---------|----------------|
| Out of memory / killed process | Instance under ~1–2 GB; models loaded at import |
| Build timeout | Large torch/sentence-transformers install; use CPU torch index |
| 502 on first request | Cold start / spin-up still loading models |
| Cookie not set / empty knowledge base after refresh | Cross-site cookie; use same-origin proxy or SameSite=None+Secure |
| CORS error | `ALLOWED_ORIGINS` missing the exact frontend origin |
| Documents vanished after restart | Non-durable disk (expected on many free tiers) |
