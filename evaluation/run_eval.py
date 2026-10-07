"""Run the golden questions through answer_question and save the raw results.

No judging and no metrics. Start it from the repo root:

    python evaluation/run_eval.py

Set EVAL_SESSION_ID to the session_id stored on that document's Chroma chunks.
Dense search filters on it. BM25 does not, so a wrong value still returns sparse hits.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# answer_question looks up ./chroma_db and bm25_{document_id}.pkl in the
# process working directory. Importing this file as a script puts evaluation/
# on sys.path, not the repo root, so fix both before importing the pipeline.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from month1_rag_engine.main import maxChunkLength, overlap
from month2_api_service.services.rag import (
    CHUNKS_PASSED_TO_LLM,
    DENSE_TOP_K,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    RRF_TOP_N,
    SPARSE_TOP_K,
    answer_question,
)

DATASET_PATH = Path("evaluation/golden_dataset.json")
RESULTS_DIR = Path("evaluation/results")

# Names in the run file match the constants they were imported from.
PIPELINE_CONSTANTS = {
    "EMBEDDING_MODEL": EMBEDDING_MODEL,
    "RERANKER_MODEL": RERANKER_MODEL,
    "DENSE_TOP_K": DENSE_TOP_K,
    "SPARSE_TOP_K": SPARSE_TOP_K,
    "RRF_TOP_N": RRF_TOP_N,
    "CHUNKS_PASSED_TO_LLM": CHUNKS_PASSED_TO_LLM,
    "maxChunkLength": maxChunkLength,
    "overlap": overlap,
}

DENSE_STOP_MESSAGE = (
    "Dense retrieval returned nothing. EVAL_SESSION_ID is probably wrong, "
    "so this run would measure BM25 only."
)


def git_snapshot():
    """Commit hash, and whether `git status --porcelain` printed anything."""
    commit = "unknown"
    dirty = True
    try:
        commit_run = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_run.returncode == 0 and commit_run.stdout.strip():
            commit = commit_run.stdout.strip()

        status_run = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if status_run.returncode == 0:
            dirty = bool(status_run.stdout.strip())
    except FileNotFoundError:
        pass
    return commit, dirty


def results_path(stamp):
    """Pick evaluation/results/run_<stamp>.json. Never overwrite an existing file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    candidate = RESULTS_DIR / f"run_{stamp}.json"
    suffix = 2
    while candidate.exists():
        candidate = RESULTS_DIR / f"run_{stamp}_{suffix}.json"
        suffix += 1
    return candidate


def load_dataset():
    with DATASET_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data["document_id"], data["items"]


def project_context_chunks(chunks):
    """Keep the rank order answer_question already returned."""
    projected = []
    for chunk in chunks:
        projected.append(
            {
                "sectionNumber": chunk["sectionNumber"],
                "chunkNumber": chunk["chunkNumber"],
                "score": chunk["score"],
                "chunk_text": chunk["chunk_text"],
            }
        )
    return projected


def question_record(
    item,
    latency,
    answer,
    context_chunks,
    citations,
    from_cache,
    dense_chunks_returned,
    sparse_chunks_returned,
    error,
):
    return {
        "id": item["id"],
        "question": item["question"],
        "type": item["type"],
        "answerable": item["answerable"],
        "answer": answer,
        "context_chunks": context_chunks,
        "citations": citations,
        "from_cache": from_cache,
        "dense_chunks_returned": dense_chunks_returned,
        "sparse_chunks_returned": sparse_chunks_returned,
        "latency_seconds": latency,
        "error": error,
        "timestamp": datetime.now().astimezone().isoformat(),
    }


def run_config(call_config):
    """First successful call's config, plus the imported pipeline constants.

    CHUNKS_PASSED_TO_LLM is the cap from code. chunks_passed_to_llm on the
    call config is how many chunks that call actually passed.
    """
    config = {}
    if call_config is not None:
        config.update(call_config)
    config.update(PIPELINE_CONSTANTS)
    return config


def run_questions(items, document_id, client, session_id, delay):
    """Execute items. Stop after 3 consecutive errors, or after a first success with no dense hits."""
    results = []
    streak = []
    first_config = None

    for index, item in enumerate(items):
        started = time.perf_counter()
        call_config = None
        try:
            result = answer_question(
                item["question"],
                document_id,
                client,
                session_id,
                temperature=0,
                use_cache=False,
            )
            latency = round(time.perf_counter() - started, 6)
            call_config = dict(result["config"])
            record = question_record(
                item,
                latency,
                result["answer"],
                project_context_chunks(result["context_chunks"]),
                result["citations"],
                result["from_cache"],
                call_config["dense_chunks_returned"],
                call_config["sparse_chunks_returned"],
                None,
            )
        except Exception as exc:
            latency = round(time.perf_counter() - started, 6)
            message = str(exc) or exc.__class__.__name__
            record = question_record(
                item,
                latency,
                None,
                None,
                None,
                None,
                None,
                None,
                message,
            )

        results.append(record)

        if call_config is None:
            # A later success clears this list, so only a run of failures counts.
            streak.append(record["error"])
            if len(streak) >= 3:
                print("Aborting after 3 consecutive failures:")
                for message in streak:
                    print(message)
                break
        else:
            streak = []
            if first_config is None:
                first_config = call_config
                if call_config["dense_chunks_returned"] == 0:
                    print(DENSE_STOP_MESSAGE)
                    break

        if delay > 0 and index + 1 < len(items):
            time.sleep(delay)

    return results, first_config


def write_run(path, dataset_question_count, results, first_config):
    commit, dirty = git_snapshot()
    payload = {
        "run_id": path.stem,
        "dataset_path": DATASET_PATH.as_posix(),
        "dataset_question_count": dataset_question_count,
        "git_commit": commit,
        "git_dirty": dirty,
        "config": run_config(first_config),
        "results": results,
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def print_summary(path, results):
    errors = sum(row["error"] is not None for row in results)
    print(f"Questions run: {len(results)}")
    print(f"Errors: {errors}")
    if results:
        average = sum(row["latency_seconds"] for row in results) / len(results)
        print(f"Average latency: {average:.3f} seconds")
    else:
        print("Average latency: n/a")
    print(f"Wrote {path.as_posix()}")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Execute the golden questions and save raw results."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N questions.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0,
        help="Seconds to sleep between questions (default 0).",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be 0 or greater")
    if args.delay < 0:
        parser.error("--delay must be 0 or greater")
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    session_id = (os.getenv("EVAL_SESSION_ID") or "").strip()
    if not api_key:
        raise SystemExit(
            "GROQ_API_KEY not found. Set it in a .env file at the repo root."
        )
    if not session_id:
        raise SystemExit(
            "EVAL_SESSION_ID is missing. Dense retrieval filters Chroma by "
            "session_id and document_id. Read session_id from the documents "
            "collection metadata for this document, then set EVAL_SESSION_ID."
        )

    client = Groq(api_key=api_key)
    document_id, items = load_dataset()
    dataset_question_count = len(items)
    if args.limit is not None:
        items = items[: args.limit]

    results, first_config = run_questions(
        items,
        document_id,
        client,
        session_id,
        args.delay,
    )
    path = results_path(datetime.now().strftime("%Y%m%d_%H%M%S"))
    write_run(path, dataset_question_count, results, first_config)
    print_summary(path, results)


if __name__ == "__main__":
    main()
