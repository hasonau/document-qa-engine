# document-qa-engine

An AI Document Assistant built as a learning project: upload PDFs, retrieve relevant chunks with hybrid search and reranking, and answer questions grounded in those chunks. It ships as a FastAPI backend plus a React chat UI that you run locally.

Layout: `month1_rag_engine/` (notebooks, CLI-era helpers, shared extraction/chunking), `month2_api_service/` (HTTP API), `month2_frontend/` (Vite + React), `evaluation/` (offline harness), `docs/DEPLOY.md` (deploy-later notes), `learning/` (curriculum).

---

## Demo

There is no hosted live demo. Run the API and frontend locally (see Quick start), then try this 60–90 second script:

1. Open the chat UI and click **Load sample document** (bundled two-page PDF about how RAG works).
2. Confirm the document appears in the knowledge base (expect more than three chunks).
3. Click one suggested question (or paste it) and wait for an answer with Sources.
4. Optionally ask a second suggested question, then delete the sample from the knowledge base.

Video walkthrough (placeholder): _add a public link here when you record one._

For hosting later (memory needs, cookies, smoke checks), see [`docs/DEPLOY.md`](docs/DEPLOY.md). **The app is not deployed.**

---

## Features

- Multi-document PDF upload with a per-session knowledge base (list, select, delete, re-upload)
- **Load sample document** (`POST /load-sample`) for a bundled original PDF plus three suggested questions in the UI
- Session cookie isolation (`session_id`); retrieval filtered by session and document
- Structure-aware chunking (heading detection, or a synthetic `"Full Document"` section)
- Hybrid retrieval: dense (Chroma) + BM25, fused with RRF, then cross-encoder rerank; top chunks to Groq
- Streaming SSE on cache hits; JSON `answer` + `citations` on cache misses
- Sources UI (heading, pages, snippet, expandable passage) and render-only answer formatting
- Upload/query limits, rate limits, filename sanitizing, generic client errors with request ids
- Offline evaluation harness under `evaluation/` (retrieval metrics; no LLM judge yet)

---

## Quick start

**Prerequisites:** developed and tested on Python 3.14.x; other versions are untested. Node.js for the frontend, and a Groq API key for live answers.

1. From the repo root, create a virtualenv, install dependencies, and copy environment variables:

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set at least:

```
GROQ_API_KEY=your_groq_api_key_here
```

Optional limits and CORS defaults are listed (commented) in `.env.example`. Never commit `.env`.

2. Start the API from the **repo root** (paths for `chroma_db/`, `bm25_*.pkl`, and uploads are relative to the working directory):

```bash
python -m uvicorn month2_api_service.main:app --reload --host 0.0.0.0 --port 8000
```

Check `GET http://localhost:8000/healthz` → `{"status":"ok"}`.

3. Start the frontend:

```bash
cd month2_frontend
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). Load the sample or upload a PDF, then ask a question.

> On Windows, if the repo path contains `&`, prefer `node .\node_modules\vite\bin\vite.js` (already used by `npm run dev`) and lint with `node .\node_modules\oxlint\bin\oxlint` instead of `npm run lint`.

### Tests and evaluation

```powershell
# API unit tests
.\.venv\Scripts\python.exe -m unittest month2_api_service.tests.test_security_helpers -v

# Frontend
cd month2_frontend
npm run build
node --test src/utils/*.test.js
node .\node_modules\oxlint\bin\oxlint

# Offline eval (needs GROQ_API_KEY and EVAL_SESSION_ID for the golden document)
cd ..
$env:EVAL_SESSION_ID = "<session_id from Chroma metadata for the golden document>"
.\.venv\Scripts\python.exe evaluation\run_eval.py --limit 3
.\.venv\Scripts\python.exe evaluation\retrieval_metrics.py
```

`EVAL_SESSION_ID` must match the golden document’s stored `session_id`. Do not commit real session ids. Results under `evaluation/results/` are gitignored.

---

## API endpoints

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/healthz` | Liveness |
| `GET` | `/` | Simple hello payload |
| `POST` | `/upload-document` | Multipart field `document`; may set `session_id` cookie |
| `POST` | `/load-sample` | Ingests the bundled sample into the current session (same limits/rate limit as upload) |
| `GET` | `/documents` | List documents for this session (401 without cookie) |
| `DELETE` | `/documents/{document_id}` | Hard-delete for this session only; foreign/missing id → 404 |
| `POST` | `/query` | Body `{ "query", "document_id" }`; cookie required; SSE or JSON |

Ingestion writes Chroma (`./chroma_db`, collection `documents`) and `bm25_{document_id}.pkl`. Query path: dense top 10 + BM25 top 10 → RRF (`k=60`, top 10) → cross-encoder → top 3 chunks into Groq (`ANSWER_MODEL` / `EXPANSION_MODEL` in `month2_api_service/services/rag.py`).

---

## Evaluation summary

Offline harness under `evaluation/`. Corpus: **one document, 9 chunks** (18 answerable + 2 unanswerable questions in `golden_dataset.json`). Metrics are therefore **saturated** (Hit@k can reach 1.0 easily) and cannot show whether reranking or query expansion helps.

Recorded baselines (from prior runs documented here; result JSON files are gitignored):

- **Day 23** — `run_20261007_103939`: of 20 questions, 2 answers failed (items 17 and 18). Retrieval on 16 answerable rows without errors: Hit@1 0.938, Hit@k 1.000, recall 1.000, MRR 0.969; both unanswerable answered `I don't know`.
- **Updated baseline (default temperature)** — `run_20261011_130503`: Hit@1 0.944, Hit@k 1.000, recall 1.000, MRR 0.972 on 18 answerable; unanswerable 2 of 2. Later `run_20261011_131637` matched those retrieval numbers with answer success 20 of 20.

The LLM judge (correctness, completeness, relevance, faithfulness) is not built. Multi-document evaluation and ablations are not done. See Decision Log and `findings/day23-eval-notes.md`.

---

## Limits and security

Configurable via environment variables (defaults in parentheses; see `.env.example`):

| Limit | Env var | Default |
| --- | --- | --- |
| Max upload size | `MAX_UPLOAD_BYTES` | 10 MB (413) |
| PDF only | magic bytes `%PDF-` | 415 otherwise |
| Max pages | `MAX_PDF_PAGES` | 100 (422) |
| Max documents per session | `MAX_DOCS_PER_SESSION` | 10 (422) |
| Min extractable text | `MIN_EXTRACTABLE_CHARS` | ~200 chars; else 422 and nothing stored |
| Max question length | `MAX_QUERY_CHARS` | 1000 (422); empty → 400 |
| `/query` rate limit | `QUERY_RATE_LIMIT_PER_MINUTE` | 20 / session / minute (429 + `Retry-After`) |
| Upload / load-sample rate limit | `UPLOAD_RATE_LIMIT_PER_MINUTE` | 5 / session / minute (429 + `Retry-After`) |
| CORS origins | `ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` |

Failed mid-ingest uploads remove the saved PDF, BM25 pickle, and Chroma chunks. Client errors are generic with a request id. Groq outages on `/query` return 503 with a fixed message. The offline eval harness calls `answer_question` directly and can see raw exceptions. The in-memory rate limiter resets on process restart and is per instance.

---

## Known limitations

- The heading heuristic over-fires on notes-style PDFs (example: **212** sections and **83** of **224** chunks under 20 words on a 40-page test PDF).
- Vague “what is in this PDF” questions retrieve arbitrary chunks on large documents.
- Cache hits show all reranked chunks as sources; fresh answers show only cited chunks.
- Answer correctness and faithfulness are not measured (no judge yet).
- The rate limiter is in memory and per instance.
- Memory footprint about **490 MB** idle working set locally (about **560 MB** after one small upload), so hosting needs roughly **1–2 GB** RAM; see `docs/DEPLOY.md`.
- Local disk (`chroma_db/`, `bm25_*.pkl`, uploaded PDFs) is ephemeral on many free hosts.

---

## Roadmap (not done)

- LLM judge for answer quality
- Multi-document evaluation and ablation studies
- Fix for heading detection on notes-style PDFs
- Hosted / durable vector store
- Deployment of the backend (deferred; memory and free-tier constraints)

---

## Month 1 — RAG Engine (Notebooks + CLI)

A Retrieval-Augmented Generation pipeline built in raw Python under `month1_rag_engine/` (no LangChain / LlamaIndex). Numbered folders are a 14-day build log; later notebooks (`12.Query_Rewriting/`, `13.Evaluation/`) are the most complete. The CLI entry point is `month1_rag_engine/main.py`. Shared sample PDF: `month1_rag_engine/data/building/`. Evaluation write-up: [`month1_rag_engine/13.Evaluation/EVALUATION.md`](month1_rag_engine/13.Evaluation/EVALUATION.md).

Month 1 uses in-memory FAISS (no persistent store in the notebooks). Setup still needs `GROQ_API_KEY` in the repo-root `.env` for generation days.

---

## Decision Log

### Index

1. [Heading detection heuristic](#1-heading-detection-heuristic)
2. [Page range trade-off for sub-chunks](#2-page-range-trade-off-for-sub-chunks)
3. [Fallback for documents with no detected headings](#3-fallback-for-documents-with-no-detected-headings)
4. [Extract the query pipeline from the route](#4-extract-the-query-pipeline-from-the-route)
5. [Evaluation design: separate stages, fixed generation settings, config per run](#5-evaluation-design-separate-stages-fixed-generation-settings-config-per-run)
6. [Named constants for pipeline settings](#6-named-constants-for-pipeline-settings)
7. [Silent BM25 fallback and dense-count guard](#7-silent-bm25-fallback-and-dense-count-guard)
8. [Saturated retrieval metrics on a 9-chunk corpus](#8-saturated-retrieval-metrics-on-a-9-chunk-corpus)
9. [Failure accounting next to metrics](#9-failure-accounting-next-to-metrics)
10. [Graceful degradation for query expansion](#10-graceful-degradation-for-query-expansion)
11. [Visible fallbacks in eval config](#11-visible-fallbacks-in-eval-config)
12. [Hard-delete order with Chroma last](#12-hard-delete-order-with-chroma-last)
13. [404 for foreign document ids](#13-404-for-foreign-document-ids)
14. [Limits and generic client errors](#14-limits-and-generic-client-errors)
15. [Render-only answer formatting](#15-render-only-answer-formatting)
16. [Not deploying to a 512 MB free host](#16-not-deploying-to-a-512-mb-free-host)

### 1. Heading detection heuristic

- **Context:** Need a cheap way to split PDF text into sections without a layout model.
- **Options:** (A) ML layout detection, (B) simple line heuristic, (C) fixed-size only.
- **Decision:** B — a line with fewer than 8 words and no trailing period is treated as a heading.
- **Consequences:** Fast and dependency-free; short non-heading lines can be misclassified (notes-style PDFs over-fragment).

### 2. Page range trade-off for sub-chunks

- **Context:** A section under one heading can split into multiple fixed-size chunks.
- **Options:** (A) track precise page per sub-chunk, (B) inherit the parent section’s startPage–endPage.
- **Decision:** B.
- **Consequences:** Simpler splitter; sub-chunks may show a wider page range than the words they contain.

### 3. Fallback for documents with no detected headings

- **Context:** Some PDFs never trip the heading heuristic.
- **Options:** (A) separate chunking path, (B) one synthetic `"Full Document"` section through the same `chunk_sections()`.
- **Decision:** B.
- **Consequences:** One chunking implementation; headerless docs still ingest.

### 4. Extract the query pipeline from the route

- **Context:** Preparing an offline evaluation harness showed that `ask` only receives chunks that are already reranked. Retrieve, rerank, and the call to `ask` lived inside `ask_question` in `month2_api_service/routes/api.py`. Rerank scores were dropped before the HTTP response, and the route required a session cookie. A script could not run or observe that pipeline.
- **Options considered:** (A) extract retrieve → rerank → ask into one service function, (B) copy the retrieval logic into the evaluation runner, (C) call the HTTP route from the runner.
- **Decision:** A. `answer_question` in `month2_api_service/services/rag.py` is the pipeline. The route only checks the session cookie and shapes the existing HTTP response.
- **Rejected:** B would make a second copy of retrieval that can drift from production. C keeps the pipeline behind a cookie and the old response, which still dropped scores and did not expose the exact chunks passed to the LLM.
- **Consequences:** The route and a future runner share one path. `context_chunks` carries section number, chunk number, rerank score, and full text for every chunk passed to the LLM, in rank order. The HTTP body is unchanged. Cache writes still happen inside `ask`. Scores from a cache hit are not available, because the cache never stored them.

### 5. Evaluation design: separate stages, fixed generation settings, config per run

- **Context:** Offline evaluation needs raw answers and the exact chunks passed to the LLM, without going through HTTP or the query cache.
- **Options considered:** (A) one script that runs questions, scores retrieval, and judges answers together; (B) separate runner, metrics, and (later) judge stages; (C) call the HTTP `/query` route from the eval script.
- **Decision:** B. `evaluation/run_eval.py` only executes questions and writes a run file. `evaluation/retrieval_metrics.py` scores retrieval offline. Rubrics live in `evaluation/rubrics.json`; the LLM judge is not built yet. The runner calls `answer_question` with `temperature=0` and `use_cache=False`, and each run file stores a config snapshot plus `git_commit` / `git_dirty`.
- **Rejected:** A mixes failure modes and makes partial re-runs harder. C keeps session cookies and response shaping in the path and still drops scores on the HTTP body.
- **Consequences:** Retrieval can be scored without a judge. Generation settings for a baseline run are fixed and recorded. Failed calls and unanswerable checks stay visible next to the metrics.

### 6. Named constants for pipeline settings

- **Context:** Top-k values, model names, and chunk size lived as literals or were only partly exposed, so a run file could not record what the pipeline actually used without guessing.
- **Options considered:** (A) leave literals and write `"unknown"` in the run config; (B) name the settings as importable constants and read them into the run file; (C) copy numbers into the runner by hand.
- **Decision:** B. `month2_api_service/services/rag.py` and `month1_rag_engine/main.py` expose named constants. The runner imports those values into the run config.
- **Rejected:** A loses reproducibility. C can drift from production.
- **Consequences:** Each run file records the settings that were imported from code. Changing a constant changes both the pipeline and the next run's config.

### 7. Silent BM25 fallback and dense-count guard

- **Context:** Dense Chroma queries filter on `session_id` and `document_id`. BM25 loads `bm25_{document_id}.pkl` and does not use `session_id`. A wrong `EVAL_SESSION_ID` therefore returns no dense hits while sparse retrieval still works.
- **Options considered:** (A) treat empty dense results as a hard error inside `query_chromadb`; (B) record per-retriever counts on the eval config and abort the run if the first success has `dense_chunks_returned == 0`; (C) ignore the mismatch and score hybrid as usual.
- **Decision:** B for the offline runner. Production retrieval behavior is unchanged.
- **Rejected:** A would change production query behavior. C would publish hybrid metrics for a BM25-only path.
- **Consequences:** A bad session id fails the eval run early. Hybrid metrics are not claimed when dense retrieval returned nothing.

### 8. Saturated retrieval metrics on a 9-chunk corpus

- **Context:** Baseline retrieval metrics on the golden document reached Hit@k 1.000, Hit@1 0.938, recall 1.000, and MRR 0.969 (16 answerable questions; chance Hit@k 0.333 with N=9 and k=3).
- **Options considered:** (A) treat these numbers as proof that reranking and query expansion help; (B) report them as a baseline and note that the metric is saturated on this corpus; (C) drop Hit@k because it cannot move.
- **Decision:** B. Keep the metrics, publish the chance baseline, and plan a larger multi-document eval set where misses remain possible.
- **Rejected:** A overclaims. C throws away a useful check for total retrieval failure.
- **Consequences:** This run cannot show whether reranking or query expansion improved ranking. Multi-document evaluation is not done yet.

### 9. Failure accounting next to metrics

- **Context:** Items 17 and 18 failed with Groq `json_validate_failed` during Day 23. Excluding them from retrieval averages without saying so would make the harness look cleaner than the run. (Day 24 later showed those labelled failures were at query expansion, before retrieval. The runner still stores nulls on any raised exception.)
- **Options considered:** (A) drop failed rows quietly from all reporting; (B) exclude them from retrieval means when `context_chunks` are absent, but list them as exclusions with the error text, and keep the answer-failure count visible; (C) retry until every row succeeds before scoring.
- **Decision:** B. Retrieval metrics are computed only over answerable rows without errors. Exclusions and the raw run's error fields remain part of the report. Day 24 also prints `Answer success: X of Y` next to the metrics.
- **Rejected:** A hides reliability problems. C blocks scoring when Groq rejects a structured output.
- **Consequences:** Hit@k 1.0 does not mean 20/20 answers succeeded. Failed answers stay visible beside the retrieval totals.

### 10. Graceful degradation for query expansion

- **Context:** Labelled re-runs showed intermittent Groq `json_validate_failed` on the optional `query_expansion` call, not on the answer call. Raising aborted the whole question even though retrieval could still use the original query.
- **Options considered:** (A) do nothing; (B) retry once then still raise; (C) retry once then fall back to the original query only; (D) change temperature/prompts/models/schemas. Option C with visible flags was chosen for simplicity and low risk when a pick was needed under time pressure.
- **Decision:** C. `_query_expansion_with_retry` tries twice (~1s sleep), then returns no expansions; `answer_question` continues. `ask` failures still raise. Production temperature, prompts, models, and schemas unchanged. Success HTTP body of `POST /query` unchanged.
- **Rejected:** A leaves flaky 500s for an optional step. D was out of scope without a confirmed root cause. Changing ask behavior was rejected (0 of 16 labelled failures were at `ask`).
- **Consequences:** Users may get an answer without expanded queries after a double expansion failure. Stub-verified only; live confirmation is watching `expansion_failed` / `expansion_attempts` on future runs. A clean 20/20 does not prove the path fired.

### 11. Visible fallbacks in eval config

- **Context:** A silent fallback would hide that retrieval ran without expansion and make metrics hard to interpret.
- **Options considered:** (A) fail soft with no signal; (B) log only; (C) record `expansion_failed` and `expansion_attempts` on the call config and each eval result row, and print counts in the runner/metrics summaries.
- **Decision:** C (with logging). Cache hits record `false` / `0`. Metrics print `Ran without expansion` only when the field exists so older run files still work.
- **Rejected:** A and B alone are not enough for offline comparison across runs.
- **Consequences:** Run JSON shows when a question continued without expansion. Operators can correlate that flag with retrieval scores later.

### 12. Hard-delete order with Chroma last

- **Context:** `DELETE /documents/{document_id}` must remove cache entries, BM25, the PDF, and Chroma chunks without leaving a half-deleted document that still answers queries.
- **Options:** (A) delete Chroma first, (B) delete Chroma last after the other stages succeed, (C) best-effort unordered deletes.
- **Decision:** B — cache → BM25 → PDF → Chroma last (see `month2_api_service/routes/api.py`).
- **Consequences:** If an early stage fails, Chroma still holds chunks so retry can finish; partial progress is reported in the error body.

### 13. 404 for foreign document ids

- **Context:** List and delete must not leak or mutate another session’s documents.
- **Options:** (A) 403 with detail, (B) 404 for missing or foreign ids, (C) silent no-op.
- **Decision:** B — foreign or missing id → 404 `"Document not found"`.
- **Consequences:** Clients cannot distinguish “not yours” from “never existed”; enumeration across sessions is harder. Chosen for simplicity and low risk.

### 14. Limits and generic client errors

- **Context:** Public upload/query needed size/type/page/session caps and safe error text without leaking stack traces.
- **Options:** (A) unlimited local-dev behavior, (B) env-configurable limits plus generic 4xx/5xx with request ids, (C) WAF-only controls.
- **Decision:** B — defaults in `config.py` / `.env.example`; mid-ingest cleanup; Groq failures → 503 fixed message on `/query`.
- **Consequences:** Abuse is bounded per process; operators use request ids in logs. Rate limits are in-memory and reset on restart.

### 15. Render-only answer formatting

- **Context:** Model answers include markdown-like lists and `**bold**`; showing raw markers is hard to read, but HTML injection is unsafe.
- **Options:** (A) `dangerouslySetInnerHTML`, (B) markdown library, (C) small UI-only formatter to React text nodes.
- **Decision:** C — `month2_frontend/src/utils/formatAnswer.js` (covered by `node --test`).
- **Consequences:** Limited markdown subset; no HTML injection from answer text. Chosen for simplicity and low risk.

### 16. Not deploying to a 512 MB free host

- **Context:** A local memory audit (Windows, CPU torch, spare port, no reload) measured about **490 MB** working set after startup and about **560 MB** after one small PDF upload (no Groq call).
- **Options:** (A) deploy anyway on 512 MB, (B) defer hosting and document a ≥1–2 GB plan, (C) change models/chunking to shrink RAM immediately.
- **Decision:** B for Day 30 lite — do not host the backend now; add sample document UX and `docs/DEPLOY.md`. Chunking, retrieval, prompts, models, schema, and cache left unchanged.
- **Consequences:** Reviewers run locally. Free-tier spin-down and non-durable disk remain open issues for a future deploy.
