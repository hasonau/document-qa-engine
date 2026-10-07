p# document-qa-engine

An AI Document Assistant: a RAG system that supports multi-document PDF upload, structure-aware chunking (heading detection with a headerless-document fallback), hybrid search (dense + BM25 fused with RRF), and cross-encoder reranking. It is exposed as a FastAPI service with streaming SSE answers and a React chat UI.

```
month1_rag_engine/     notebooks + reusable RAG logic + CLI
month2_api_service/    FastAPI service (upload, query, hybrid retrieval)
month2_frontend/       Vite + React chat UI
learning/              study curriculum
```

---

## Architecture

Three layers:

**`month1_rag_engine/`** — pure, reusable RAG logic with no HTTP concerns. `extract_pages()` pulls text with pdfplumber. `detect_sections()` splits pages into heading-bounded sections. `chunk_sections()` then splits each section with fixed-size windows (200 words, 30-word overlap) and attaches `sectionNumber`, `heading`, `startPage`, `endPage`, and `chunkNumber`. Embedding and CLI-era FAISS indexing / generation helpers still live here for the Month 1 notebooks and CLI.

**`month2_api_service/`** — FastAPI service that wraps that engine:
- `main.py` — app setup, CORS for the Vite origin (`http://localhost:5173` / `http://127.0.0.1:5173`), dotenv, router mount
- `routes/api.py` — HTTP endpoints: `POST /upload-document`, `POST /query` (SSE), `GET /healthz`
- `services/rag.py` — RAG business logic: Chroma persistence, BM25 sparse query, RRF fusion, cross-encoder rerank, streaming generation

**`month2_frontend/`** — Vite + React chat UI. Upload a PDF to `/upload-document` (cookie session), then ask questions against `/query` over SSE (`credentials: "include"`). Answers stream token-by-token; sources (chunk number + page range) render under each assistant message.

Ingestion path: PDF → `extract_pages` → `detect_sections` → `chunk_sections` (or a synthetic `"Full Document"` section if no headings) → Chroma (`./chroma_db`, collection `documents`) + a per-document BM25 pickle (`bm25_{document_id}.pkl`).

Query path: session cookie required → dense Chroma query (filtered by `session_id` + `document_id`, top 10) and BM25 (top 10) → RRF (`k=60`, top 10) → cross-encoder rerank → top 3 chunks into Groq → SSE `answer` tokens, then a `sources` event.

---

## Tech stack

- **FastAPI** + **uvicorn** — HTTP service
- **ChromaDB** — persistent vector store (`PersistentClient`, path `./chroma_db`)
- **Groq** (`llama-3.3-70b-versatile`) — generation
- **sentence-transformers** (`all-MiniLM-L6-v2`) — dense embeddings
- **rank_bm25** (`BM25Okapi`) — sparse retrieval
- **cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **sse-starlette** — streaming query responses
- **pdfplumber** — PDF text extraction
- **React** (Vite) — chat UI

---

## Features implemented so far

- **Multi-document upload** — each PDF gets a `document_id`; the UI queries that id
- **Session-based tenant isolation** — upload sets a `session_id` cookie; Chroma queries are filtered by both `session_id` and `document_id` (`/query` returns 401 if the cookie is missing)
- **Persistent Chroma storage** — embeddings survive process restarts under `./chroma_db`
- **Hybrid search** — dense (Chroma) + sparse (BM25) fused with Reciprocal Rank Fusion
- **Cross-encoder reranking** — RRF candidates rescored; top 3 go to the LLM
- **Streaming SSE responses** — `/query` yields `answer` tokens, then a `sources` JSON payload
- **Structure-aware chunking** — heading detection, then per-section fixed-size splits; documents with no detected headings fall back to one synthetic `"Full Document"` section and the same `chunk_sections()` path
- **Source metadata in context** — the prompt includes section heading, page range, and chunk number
- **Offline evaluation harness** — golden set, raw run runner, and retrieval metrics under `evaluation/` (LLM judge not built yet)

---

## Setup / running

Requires Python 3.10+ and Node.js (for the frontend).

1. Install Python dependencies from the repo root:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file at the **repo root** with your Groq API key (see `.env.example`):

```
GROQ_API_KEY=your_groq_api_key_here
```

You can get a free API key at [console.groq.com](https://console.groq.com). The `.env` file is gitignored — never commit it.

3. Start the FastAPI server from the repo root:

```bash
python -m uvicorn month2_api_service.main:app --reload --host 0.0.0.0 --port 8000
```

The API listens on [http://localhost:8000](http://localhost:8000). `GET /healthz` should return `{"status": "ok"}`.

4. Start the React frontend:

```bash
cd month2_frontend
npm install
npm run dev
```

Open the URL Vite prints (usually [http://localhost:5173](http://localhost:5173)). Upload a PDF, then ask a question.

> Note: this repo path contains `&`, which breaks Windows `vite.cmd` shims. Frontend scripts call Vite via `node` instead.

Month 1 notebooks and the CLI still use their own setup (FAISS, Jupyter); see the Month 1 section below.

---

## Evaluation

Offline harness under `evaluation/`. It calls the same `answer_question` path as production (not the HTTP route), saves raw results, then scores retrieval against the golden evidence. The LLM judge is not built yet.

| File | Role |
| --- | --- |
| `evaluation/golden_dataset.json` | 20 questions on one document (18 answerable, 2 unanswerable; 9 chunks) |
| `evaluation/rubrics.json` | Four dimensions (correctness, completeness, relevance, faithfulness); judge not implemented |
| `evaluation/run_eval.py` | Runner: executes questions, writes `evaluation/results/run_<timestamp>.json` |
| `evaluation/retrieval_metrics.py` | Hit@1 / Hit@k / recall / MRR, plus unanswerable "I don't know" check |

### How to run

From the repo root, with the project venv and a `.env` that contains `GROQ_API_KEY`:

```powershell
# Dense retrieval filters Chroma by session_id + document_id.
# Read session_id from the documents collection metadata for the golden document_id.
# Do not commit the real value.
$env:EVAL_SESSION_ID = "<session_id from Chroma chunk metadata>"

.\.venv\Scripts\python.exe evaluation/run_eval.py --limit 1
.\.venv\Scripts\python.exe evaluation/run_eval.py --delay 1
.\.venv\Scripts\python.exe evaluation/retrieval_metrics.py evaluation/results/run_<timestamp>.json
```

The runner uses `temperature=0` and `use_cache=False`. Each run file stores a config snapshot (models, temperature, cache flag, chunk counts, named top-k constants), `git_commit`, and `git_dirty`. Results under `evaluation/results/` are gitignored.

### Baseline results

Run `run_20261007_103939` at commit `1e554548d56d8d851918907e811f0ea304d778bd` (`git_dirty`: false). Of 20 questions, 2 answers failed (items 17 and 18). Retrieval metrics cover the 16 answerable questions without errors:

| Metric | Value |
| --- | --- |
| Hit@1 | 0.938 |
| Hit@k | 1.000 |
| Recall | 1.000 |
| MRR | 0.969 |
| Chance Hit@1 (`1/N`) | 0.111 |
| Chance Hit@k (`k/N`) | 0.333 |

`N` = 9 chunks; mean `k` = 3. Both unanswerable questions answered `I don't know`.

### Limitations and known issues

- Metrics are saturated on this 9-chunk single-document corpus (Hit@k 1.0), so they cannot show whether reranking or query expansion helps.
- Evaluation covers one document only. Multi-document evaluation is not done.
- The LLM judge (correctness, faithfulness, and the other rubric dimensions) is not built yet.
- Items 17 and 18 failed with Groq `json_validate_failed` and an empty `failed_generation` on a strict `json_schema` call. The failing stage (answer vs query expansion) is unconfirmed. Failed rows saved no retrieved chunks, because the runner stores nulls on any exception.
- A wrong `EVAL_SESSION_ID` does not raise an error: dense search can return nothing while BM25 still returns chunks, so the pipeline becomes BM25-only. The runner records per-retriever counts and stops if the first successful call has `dense_chunks_returned == 0`.

Technical notes: `findings/day23-eval-notes.md`.

---

## Month 1 — RAG Engine (Notebooks + CLI)

A Retrieval-Augmented Generation (RAG) pipeline built **from scratch in raw Python** — no LangChain, no LlamaIndex, no framework abstractions. Every stage (PDF extraction, chunking, embeddings, vector search, prompting, generation, evaluation) is written by hand to understand the core mechanics of how RAG systems actually work.

Work lives under `month1-rag-engine/`, organized as a 14-day build log: each numbered folder is one day's work, progressively layering a new capability onto the pipeline. The final pipeline lives in the later notebooks (`12.Query_Rewriting/` and `13.Evaluation/` contain the most complete versions).

### Architecture

The end-to-end pipeline flow:

```
PDF document
    │
    ▼
1. Text extraction          pdfplumber, page by page
    │
    ▼
2. Chunking                 heading-based (hierarchical) chunks,
    │                       with page/chunk metadata (startPage, endPage, chunk id)
    ▼
3. Embeddings               sentence-transformers (all-MiniLM-L6-v2)
    │
    ▼
4. Vector index             FAISS (in-memory)
    │
    ▼
5. Retrieval                query → embedding → top-k nearest chunks
    │                       (optional: HyDE query rewriting — the LLM writes a
    │                        hypothetical answer first, and *that* is embedded
    │                        and used for the search instead of the raw query)
    ▼
6. Generation               Groq LLM (llama-3.3-70b-versatile) answers using
                            only the retrieved context, with hallucination
                            guardrails ("say I don't know if not found") and
                            self-reported context citation
```

Everything is held in memory — there is no persistent vector store. Each notebook rebuilds the index from the source PDF when run.

### Setup

Requires Python 3.10+ and Jupyter.

1. Install dependencies:

```bash
pip install pdfplumber sentence-transformers faiss-cpu numpy scikit-learn groq python-dotenv jupyter
```

2. Create a `.env` file at the **repo root** with your Groq API key (see `.env.example`):

```
GROQ_API_KEY=your_groq_api_key_here
```

You can get a free API key at [console.groq.com](https://console.groq.com). The `.env` file is gitignored — never commit it.

> Note: days 1–7 don't need an API key at all (retrieval only). The Groq key is only required from day 8 onward, when the LLM generation layer is added.

### Quick Start

Run the full pipeline from the command line (PDF path + question):

```bash
python month1-rag-engine/main.py month1-rag-engine/data/building/Muhammad-pages.pdf "What did he say when asked for protection?"
```

This runs extraction → heading-based chunking → embeddings → FAISS retrieval → Groq generation, then prints the answer with page/chunk citations. Requires `GROQ_API_KEY` in the repo-root `.env`.

### How to run

#### Notebooks

1. Start Jupyter from the repo root:

```bash
jupyter notebook
```

2. Open any day's notebook under `month1-rag-engine/` (e.g. `month1-rag-engine/13.Evaluation/retrieval_evaluation.ipynb`) and run the cells top to bottom.

Notebooks load the shared sample PDF via relative paths (e.g. `../data/building/Muhammad-pages.pdf`) and the repo-root `.env` via `../../.env`, so run them from their own folder — which Jupyter does by default when you open them in place.

For the most complete pipeline, use:

- `month1-rag-engine/12.Query_Rewriting/query_rewriting.ipynb` — full RAG loop with optional HyDE
- `month1-rag-engine/13.Evaluation/retrieval_evaluation.ipynb` — the same pipeline plus the 10-question evaluation harness

### Folder structure

Each numbered folder corresponds to one day of the build and contains that day's notebook. Later days re-include the earlier stages, so each notebook is self-contained.

| Folder | Day | What was built |
|---|---|---|
| `1.Extraction/` | 1 | PDF → raw text extraction with pdfplumber |
| `2.Chunking/` | 2 | Splitting extracted text into chunks |
| `3.Embeddings/` | 3 | Chunk → vector embeddings with sentence-transformers |
| `4.Similarity_Search/` | 4 | Manual cosine-similarity search (scikit-learn), top-k retrieval |
| `5.Faiss_Implementation/` | 5 | Replacing manual search with a FAISS index |
| `6.Retrieval Pipeline/` | 6 | Full retrieval pipeline end to end (no LLM yet) |
| `7.Review_Rebuild/` | 7 | The whole pipeline rebuilt from memory, without looking |
| `8.LLM_layer/` | 8 | Generation step added — retrieved chunks sent to Groq LLM |
| `9.Prompt Engineering & Hallucination Control/` | 9 | Context labeling, "I don't know" guardrails, self-reported citation |
| `10.Metadata/` | 10 | Page number and chunk id metadata attached to every chunk |
| `11.Hierarchical_Chunking/` | 11 | Heading-based chunking with cross-page startPage/endPage tracking |
| `12.Query_Rewriting/` | 12 | HyDE (Hypothetical Document Embeddings) query rewriting experiment |
| `13.Evaluation/` | 13 | 10-question eval, HyDE vs No-HyDE — see `13.Evaluation/EVALUATION.md` |
| — | 14 | Final polish: cleanup and documentation |

Shared Month 1 assets:

- `month1-rag-engine/data/building/` — the shared sample PDF (a short biography of Muhammad, used only as sample text for building the pipeline)
- `month1-rag-engine/main.py` — CLI entry point for the full pipeline
- `learning/` — the study curriculum this build follows (repo root)

### Key findings & limitations

Full write-up with per-question analysis in [`month1-rag-engine/13.Evaluation/EVALUATION.md`](month1-rag-engine/13.Evaluation/EVALUATION.md). Highlights:

- **HyDE underperforms on narrow, single-topic documents.** Across a 10-question eval run twice, HyDE never outright won: 7/10 were ties, 2/10 went to plain retrieval, and 1/10 both failed. On a small document there aren't enough competing chunks for a hypothetical-answer embedding to improve retrieval — the correct chunk gets found either way, so HyDE just adds an extra LLM call, extra noise, and extra run-to-run variance.
- **Correct retrieval doesn't guarantee correct generation.** In one case both pipelines retrieved exactly the right chunk, but the LLM still failed to extract the specific quoted word from it. Retrieval failures and generation failures are distinct and worth tracking separately.
- **Chunk size and strategy depend on document structure.** Fixed-size chunking (day 2) versus heading-based hierarchical chunking (day 11) produce meaningfully different retrieval behavior; there is no universally right chunk size — it follows from how the source document is organized.
- **Grounding instructions are robust to noisy input.** Even when HyDE fed a fabricated hypothetical into the search step, the "say I don't know if it's not in the context" prompt instruction held up and prevented hallucinated answers for genuinely absent information.
- **Other limitations:** the index is in-memory only (no persistence), the system handles a single document, and it has only been evaluated on one short PDF — findings may not transfer to large or multi-topic corpora.

### Tech stack

- **Python** (raw, no RAG frameworks)
- **pdfplumber** — PDF text extraction
- **sentence-transformers** (`all-MiniLM-L6-v2`) — embeddings
- **FAISS** — vector similarity search (scikit-learn cosine similarity in the early days)
- **NumPy** — vector math
- **Groq** (`llama-3.3-70b-versatile`) — LLM generation, and the HyDE hypothetical-answer step
- **python-dotenv** — API key loading
- **Jupyter** — all work is in notebooks

---

## Decision Log

**Heading detection heuristic (Day 17):** Headings are detected using a simple rule — a line with fewer than 8 words and no trailing period is treated as a heading. This avoids ML-based layout detection, keeping the pipeline fast and dependency-free, at the cost of occasionally misclassifying short non-heading lines.

**Page range trade-off for sub-chunks (Day 17):** When a section under one heading is split into multiple smaller chunks, every sub-chunk inherits the parent section's full startPage–endPage range rather than a precise individual page number. This was a deliberate simplification to avoid tracking page numbers at the line level inside the sub-chunking splitter.

**Fallback for documents with no detected headings (Day 17):** If no headings are found in a document, the whole document is wrapped into a single synthetic section (heading: "Full Document") and passed through the same `chunk_sections()` function used for heading-based documents, avoiding duplicate chunking logic.

**Extract the query pipeline from the route**

- **Context:** Preparing an offline evaluation harness showed that `ask` only receives chunks that are already reranked. Retrieve, rerank, and the call to `ask` lived inside `ask_question` in `month2_api_service/routes/api.py`. Rerank scores were dropped before the HTTP response, and the route required a session cookie. A script could not run or observe that pipeline.
- **Options considered:** (A) extract retrieve → rerank → ask into one service function, (B) copy the retrieval logic into the evaluation runner, (C) call the HTTP route from the runner.
- **Decision:** A. `answer_question` in `month2_api_service/services/rag.py` is the pipeline. The route only checks the session cookie and shapes the existing HTTP response. One function is the production path, so the code under evaluation is the code that serves requests, and a script can call it without HTTP.
- **Rejected:** B would make a second copy of retrieval that can drift from production. C keeps the pipeline behind a cookie and the old response, which still dropped scores and did not expose the exact chunks passed to the LLM.
- **Consequences:** The route and a future runner share one path. `context_chunks` carries section number, chunk number, rerank score, and full text for every chunk passed to the LLM, in rank order, so faithfulness can be checked against that context. The HTTP body is unchanged: a cache hit is still an SSE `answer` event plus a `sources` event, and a cache miss is still `{"answer", "citations"}`. Cache writes still happen inside `ask`. Scores from a cache hit are not available, because the cache never stored them.

**Evaluation design: separate stages, fixed generation settings, config per run**

- **Context:** Offline evaluation needs raw answers and the exact chunks passed to the LLM, without going through HTTP or the query cache.
- **Options considered:** (A) one script that runs questions, scores retrieval, and judges answers together; (B) separate runner, metrics, and (later) judge stages; (C) call the HTTP `/query` route from the eval script.
- **Decision:** B. `evaluation/run_eval.py` only executes questions and writes a run file. `evaluation/retrieval_metrics.py` scores retrieval offline. Rubrics live in `evaluation/rubrics.json`; the LLM judge is not built yet. The runner calls `answer_question` with `temperature=0` and `use_cache=False`, and each run file stores a config snapshot plus `git_commit` / `git_dirty`.
- **Rejected:** A mixes failure modes and makes partial re-runs harder. C keeps session cookies and response shaping in the path and still drops scores on the HTTP body.
- **Consequences:** Retrieval can be scored without a judge. Generation settings for a baseline run are fixed and recorded. Failed calls and unanswerable checks stay visible next to the metrics.

**Named constants for pipeline settings**

- **Context:** Top-k values, model names, and chunk size lived as literals or were only partly exposed, so a run file could not record what the pipeline actually used without guessing.
- **Options considered:** (A) leave literals and write `"unknown"` in the run config; (B) name the settings as importable constants and read them into the run file; (C) copy numbers into the runner by hand.
- **Decision:** B. `month2_api_service/services/rag.py` and `month1_rag_engine/main.py` expose named constants (`EMBEDDING_MODEL`, `RERANKER_MODEL`, `DENSE_TOP_K`, `SPARSE_TOP_K`, `RRF_TOP_N`, `CHUNKS_PASSED_TO_LLM`, `maxChunkLength`, `overlap`). The runner imports those values into the run config.
- **Rejected:** A loses reproducibility. C can drift from production.
- **Consequences:** Each run file records the settings that were imported from code. Changing a constant changes both the pipeline and the next run's config.

**Silent BM25 fallback and dense-count guard**

- **Context:** Dense Chroma queries filter on `session_id` and `document_id`. BM25 loads `bm25_{document_id}.pkl` and does not use `session_id`. A wrong `EVAL_SESSION_ID` therefore returns no dense hits while sparse retrieval still works.
- **Options considered:** (A) treat empty dense results as a hard error inside `query_chromadb`; (B) record per-retriever counts on the eval config and abort the run if the first success has `dense_chunks_returned == 0`; (C) ignore the mismatch and score hybrid as usual.
- **Decision:** B for the offline runner. Production retrieval behavior is unchanged. The runner prints that the session id is probably wrong and that the run would measure BM25 only.
- **Rejected:** A would change production query behavior. C would publish hybrid metrics for a BM25-only path.
- **Consequences:** A bad session id fails the eval run early. Hybrid metrics are not claimed when dense retrieval returned nothing.

**Saturated retrieval metrics on a 9-chunk corpus**

- **Context:** Baseline retrieval metrics on the golden document reached Hit@k 1.000, Hit@1 0.938, recall 1.000, and MRR 0.969 (16 answerable questions; chance Hit@k 0.333 with N=9 and k=3).
- **Options considered:** (A) treat these numbers as proof that reranking and query expansion help; (B) report them as a baseline and note that the metric is saturated on this corpus; (C) drop Hit@k because it cannot move.
- **Decision:** B. Keep the metrics, publish the chance baseline, and plan a larger multi-document eval set where misses remain possible.
- **Rejected:** A overclaims. C throws away a useful check for total retrieval failure.
- **Consequences:** This run cannot show whether reranking or query expansion improved ranking. Multi-document evaluation is not done yet.

**Failure accounting next to metrics**

- **Context:** Items 17 and 18 failed during generation (Groq `json_validate_failed`). Excluding them from retrieval averages without saying so would make the harness look cleaner than the run.
- **Options considered:** (A) drop failed rows quietly from all reporting; (B) exclude them from retrieval means (no saved `context_chunks`) but list them as exclusions with the error text, and keep the 2/20 answer-failure count in the run summary; (C) retry until every row succeeds before scoring.
- **Decision:** B. Retrieval metrics are computed only over answerable rows without errors. Exclusions and the raw run's error fields remain part of the report.
- **Rejected:** A hides reliability problems. C blocks scoring when Groq rejects a structured output.
- **Consequences:** Hit@k 1.0 does not mean 20/20 answers succeeded. Failed answers stay visible beside the retrieval totals.
