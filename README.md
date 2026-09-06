# document-qa-engine

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
