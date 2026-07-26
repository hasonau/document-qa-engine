# document-qa-engine

A Retrieval-Augmented Generation (RAG) pipeline built **from scratch in raw Python** — no LangChain, no LlamaIndex, no framework abstractions. Every stage (PDF extraction, chunking, embeddings, vector search, prompting, generation, evaluation) is written by hand to understand the core mechanics of how RAG systems actually work.

The repo is organized as a 14-day build log: each numbered folder is one day's work, progressively layering a new capability onto the pipeline. The final pipeline lives in the later notebooks (`12.Query_Rewriting/` and `13.Evaluation/` contain the most complete versions).

## Architecture

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

## Setup

Requires Python 3.10+ and Jupyter.

1. Install dependencies:

```bash
pip install pdfplumber sentence-transformers faiss-cpu numpy scikit-learn groq python-dotenv jupyter
```

2. Create a `.env` file at the repo root with your Groq API key (see `.env.example`):

```
GROQ_API_KEY=your_groq_api_key_here
```

You can get a free API key at [console.groq.com](https://console.groq.com). The `.env` file is gitignored — never commit it.

> Note: days 1–7 don't need an API key at all (retrieval only). The Groq key is only required from day 8 onward, when the LLM generation layer is added.

## Quick Start

Run the full pipeline from the command line (PDF path + question):

```bash
python main.py data/building/Muhammad-pages.pdf "What did he say when asked for protection?"
```

This runs extraction → heading-based chunking → embeddings → FAISS retrieval → Groq generation, then prints the answer with page/chunk citations. Requires `GROQ_API_KEY` in `.env`.

## How to run

### Notebooks

1. Start Jupyter from the repo root:

```bash
jupyter notebook
```

2. Open any day's notebook (e.g. `13.Evaluation/retrieval_evaluation.ipynb`) and run the cells top to bottom.

Notebooks load the shared sample PDF via relative paths (e.g. `../data/building/Muhammad-pages.pdf`) and the `.env` from the repo root (`../.env`), so run them from their own folder — which Jupyter does by default when you open them in place.

For the most complete pipeline, use:

- `12.Query_Rewriting/query_rewriting.ipynb` — full RAG loop with optional HyDE
- `13.Evaluation/retrieval_evaluation.ipynb` — the same pipeline plus the 10-question evaluation harness

## Folder structure

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
| — | 14 | Final polish: cleanup and this documentation |

Shared assets live at the repo root:

- `data/building/` — the shared sample PDF (a short biography of Muhammad, used only as sample text for building the pipeline)
- `learning/` — the study curriculum this build follows

## Key findings & limitations

Full write-up with per-question analysis in [`13.Evaluation/EVALUATION.md`](13.Evaluation/EVALUATION.md). Highlights:

- **HyDE underperforms on narrow, single-topic documents.** Across a 10-question eval run twice, HyDE never outright won: 7/10 were ties, 2/10 went to plain retrieval, and 1/10 both failed. On a small document there aren't enough competing chunks for a hypothetical-answer embedding to improve retrieval — the correct chunk gets found either way, so HyDE just adds an extra LLM call, extra noise, and extra run-to-run variance.
- **Correct retrieval doesn't guarantee correct generation.** In one case both pipelines retrieved exactly the right chunk, but the LLM still failed to extract the specific quoted word from it. Retrieval failures and generation failures are distinct and worth tracking separately.
- **Chunk size and strategy depend on document structure.** Fixed-size chunking (day 2) versus heading-based hierarchical chunking (day 11) produce meaningfully different retrieval behavior; there is no universally right chunk size — it follows from how the source document is organized.
- **Grounding instructions are robust to noisy input.** Even when HyDE fed a fabricated hypothetical into the search step, the "say I don't know if it's not in the context" prompt instruction held up and prevented hallucinated answers for genuinely absent information.
- **Other limitations:** the index is in-memory only (no persistence), the system handles a single document, and it has only been evaluated on one short PDF — findings may not transfer to large or multi-topic corpora.

## Tech stack

- **Python** (raw, no RAG frameworks)
- **pdfplumber** — PDF text extraction
- **sentence-transformers** (`all-MiniLM-L6-v2`) — embeddings
- **FAISS** — vector similarity search (scikit-learn cosine similarity in the early days)
- **NumPy** — vector math
- **Groq** (`llama-3.3-70b-versatile`) — LLM generation, and the HyDE hypothetical-answer step
- **python-dotenv** — API key loading
- **Jupyter** — all work is in notebooks
