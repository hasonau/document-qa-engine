# Day 23 evaluation notes

Source: `evaluation/` harness files, baseline run `evaluation/results/run_20261007_103939.json`, and metrics `evaluation/results/run_20261007_103939_retrieval.json`. RAG pipeline code was not changed for this note.

## Harness design

Offline evaluation is split into stages:

- **Dataset** — `evaluation/golden_dataset.json`: 20 questions on one document (18 answerable, 2 unanswerable; 9 chunks).
- **Runner** — `evaluation/run_eval.py` calls `answer_question` directly (not HTTP) with `temperature=0` and `use_cache=False`. Each run writes a new `evaluation/results/run_<timestamp>.json` with answers, `context_chunks` (chunks passed to the LLM, in rank order), per-call config, `dense_chunks_returned`, `sparse_chunks_returned`, latency, and errors.
- **Retrieval metrics** — `evaluation/retrieval_metrics.py` scores Hit@1, Hit@k, recall, and MRR against golden evidence (id match or quote containment). It also checks unanswerable answers for exact "I don't know". No LLM.
- **Rubrics** — `evaluation/rubrics.json` defines four dimensions (correctness, completeness, relevance, faithfulness). The LLM judge is not built yet.

The run file records a config snapshot from the first successful call plus named pipeline constants (`EMBEDDING_MODEL`, `RERANKER_MODEL`, top-k values, `CHUNKS_PASSED_TO_LLM`, `maxChunkLength`, `overlap`), plus `git_commit` and `git_dirty`.

## Baseline run

| Field | Value |
| --- | --- |
| `run_id` | `run_20261007_103939` |
| `git_commit` | `1e554548d56d8d851918907e811f0ea304d778bd` |
| `git_dirty` | `false` |

Retrieval metrics (16 answerable questions with no error):

| Metric | Value |
| --- | --- |
| Hit@1 | 0.938 |
| Hit@k | 1.000 |
| Recall | 1.000 |
| MRR | 0.969 |
| Chance Hit@1 (`1/N`) | 0.111 |
| Chance Hit@k (`k/N`) | 0.333 |

`N` = 9 chunks in the document; mean `k` = 3 chunks passed to the LLM.

Both unanswerable questions (ids 19 and 20) answered `I don't know`. 2 of 20 answers failed (items 17 and 18).

## Findings

1. **Silent BM25-only path.** A wrong `EVAL_SESSION_ID` does not raise. Dense search is filtered by `session_id` and `document_id`, so it can return nothing while BM25 (keyed only by `document_id`) still returns chunks. The pipeline continues as BM25-only. The runner records `dense_chunks_returned` and `sparse_chunks_returned` and aborts if the first successful call has `dense_chunks_returned == 0`.

2. **Generation failures drop retrieval.** Items 17 and 18 failed with Groq HTTP 400 `json_validate_failed` and an empty `failed_generation`, from a strict `json_schema` call. The failing stage is unconfirmed (answer call or query-expansion call). On any exception the runner stores nulls for answer fields, including `context_chunks`, so those rows have no saved retrieval data.

## Limitation

Hit@k is 1.0 on this 9-chunk single-document corpus. The retrieval metrics are saturated and cannot show whether reranking or query expansion helps. Evaluation so far covers one document only, not multi-document.

## Not done yet

- LLM judge (correctness, faithfulness, and the other rubric dimensions)
- Ablation runs
- Multi-document eval set
- Runner option to re-run specific question ids
- Separating retrieval from generation so a generation failure keeps its retrieved chunks
- Root-causing items 17 and 18
