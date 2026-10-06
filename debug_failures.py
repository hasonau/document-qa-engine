"""Read query logs and run a wrong-chunk generation check.

This script does not change retrieval, reranking, or generation. It only
reads logs, the on-disk query cache, and the BM25 chunk file, then calls
the same Groq chat completion the pipeline uses. It does not call ask(),
because ask() writes the query cache and would replace a real answer.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

ROOT = Path(__file__).resolve().parent
LOG_FILE = ROOT / "logs" / "query_logs.jsonl"
CHROMA_DB = ROOT / "chroma_db" / "chroma.sqlite3"

# Same instruction string and model as month2_api_service/services/rag.py ask().
INSTRUCTIONS = (
    "\nAnswer the question using only the provided context. "
    "If the context does not contain enough information, respond exactly with 'I don't know'. "
    "If the answer is found, cite the source number, page number, and chunk number used."
)
GENERATION_MODEL = "openai/gpt-oss-120b"

# Real logged questions. Each preferred heading is a section that was not
# in the chunks actually retrieved for that query.
SUITE = [
    {
        "label": "Ta'if",
        "query": "What happened when Muhammad tried to preach in Taif?",
        "prefer_heading": "Social exclusion of the Banu Hashim",
    },
    {
        "label": "Abu Talib",
        "query": "How did Abu Talib respond when Muhammad said he would not stop preaching?",
        "prefer_heading": "Migration to Abyssinia",
    },
    {
        "label": "Abyssinia",
        "query": "Why did the early Muslims migrate to Abyssinia?",
        "prefer_heading": "Quraysh delegation to Yathrib",
    },
]


class _Bm25Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


class _ChunkUnpickler(pickle.Unpickler):
    """Load bm25_*.pkl without importing rank_bm25. Only the chunk list is used."""

    def find_class(self, module, name):
        if module.startswith("rank_bm25"):
            return _Bm25Stub
        return super().find_class(module, name)


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_logs(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Query log not found: {path}")
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                entries.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path} line {line_number}") from exc
    return entries


def load_cache_sources(db_path: Path) -> dict[tuple[str, str], list[dict]]:
    """Map (document_id, query) to the context chunks stored by ask()."""
    if not db_path.exists():
        return {}

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT id, key, string_value FROM embedding_metadata WHERE key IN ('document_id', 'query', 'sources')"
        ).fetchall()
    finally:
        connection.close()

    grouped: dict[int, dict[str, str]] = {}
    for row_id, key, value in rows:
        grouped.setdefault(row_id, {})[key] = value

    cache: dict[tuple[str, str], list[dict]] = {}
    for meta in grouped.values():
        document_id = meta.get("document_id")
        query = meta.get("query")
        raw_sources = meta.get("sources")
        if not document_id or not query or not raw_sources:
            continue
        try:
            sources = json.loads(raw_sources)
        except json.JSONDecodeError:
            continue
        if isinstance(sources, list):
            cache[(document_id, query)] = sources
    return cache


def load_chunks(document_id: str) -> list[dict]:
    pickle_path = ROOT / f"bm25_{document_id}.pkl"
    if not pickle_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {pickle_path}")
    with pickle_path.open("rb") as handle:
        payload = _ChunkUnpickler(handle).load()
    chunks = payload["chunks"]
    if not chunks:
        raise ValueError(f"No chunks stored for document {document_id}")
    return chunks


def excerpt(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def format_pages(chunk: dict) -> str:
    start = chunk.get("startPage")
    end = chunk.get("endPage")
    if start is None:
        return "unknown"
    if end is not None and end != start:
        return f"{start}-{end}"
    return str(start)


def print_logs(entries: list[dict], cache: dict[tuple[str, str], list[dict]]) -> None:
    print("=" * 72)
    print(f"QUERY LOGS  ({len(entries)} entries from {LOG_FILE.relative_to(ROOT)})")
    print("Citations are not stored in the log file.")
    print("Each citation list below is the context saved in the query cache for that query.")
    print("=" * 72)

    for index, entry in enumerate(entries, start=1):
        query = entry.get("query", "")
        document_id = entry.get("document_id", "")
        citations = entry.get("citations")
        if not isinstance(citations, list):
            citations = cache.get((document_id, query))

        print()
        print(f"--- {index}. {entry.get('timestamp', 'no timestamp')} ---")
        print(f"query: {query}")
        print(f"document_id: {document_id}")
        print(f"success: {entry.get('success')}")
        print(f"answer: {entry.get('answer', '')}")
        if not citations:
            print("citations: none recorded")
            continue
        print(f"citations: {len(citations)}")
        for citation_index, chunk in enumerate(citations, start=1):
            if not isinstance(chunk, dict):
                print(f"  [{citation_index}] {chunk}")
                continue
            heading = chunk.get("heading", "unknown section")
            chunk_number = chunk.get("chunkNumber", "?")
            print(
                f"  [{citation_index}] {heading} | chunk {chunk_number} | pages {format_pages(chunk)}"
            )
            text = chunk.get("chunk_text")
            if text:
                print(f"      {excerpt(text)}")


def _token_overlap(query: str, text: str) -> int:
    query_tokens = {token.lower() for token in query.split() if len(token) > 2}
    text_tokens = {token.lower() for token in text.split()}
    return len(query_tokens & text_tokens)


def normal_headings_for(
    query: str,
    document_id: str,
    chunks: list[dict],
    cache: dict[tuple[str, str], list[dict]],
) -> list[str]:
    cached = cache.get((document_id, query))
    if cached:
        headings = []
        for chunk in cached:
            heading = chunk.get("heading")
            if heading and heading not in headings:
                headings.append(heading)
        return headings

    best = max(chunks, key=lambda chunk: _token_overlap(query, chunk.get("chunk_text", "")))
    return [best.get("heading", "")]


def pick_wrong_chunk(
    chunks: list[dict],
    avoid_headings: list[str],
    prefer_heading: str | None = None,
) -> dict:
    avoided = set(avoid_headings)
    if prefer_heading and prefer_heading not in avoided:
        for chunk in chunks:
            if chunk.get("heading") == prefer_heading:
                return chunk

    for chunk in chunks:
        if chunk.get("heading") not in avoided:
            return chunk
    raise ValueError(
        "Every stored chunk belongs to a section that would normally be retrieved"
    )


def build_single_chunk_message(query: str, chunk: dict) -> str:
    context = "Source 1 :\n"
    context += f"\nSection: {chunk['heading']}"
    context += f"\nPage Number: {chunk['startPage']}"
    if chunk["startPage"] != chunk["endPage"]:
        context += f" - {chunk['endPage']}"
    context += f"\nChunk Number: {chunk['chunkNumber']}\n"
    context += chunk["chunk_text"] + "\n"
    return "Instructions :\n" + INSTRUCTIONS + "\n Context :" + context + "\n Question : \n" + query


def generate_from_chunk(client: Groq, query: str, chunk: dict) -> dict:
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": build_single_chunk_message(query, chunk)}],
        stream=False,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "rag_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "citations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"source": {"type": "integer"}},
                                "required": ["source"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["answer", "citations"],
                    "additionalProperties": False,
                },
            },
        },
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def refusal_verdict(answer: str) -> str:
    normalized = " ".join(answer.strip().split()).rstrip(".").lower().replace("’", "'")
    if normalized == "i don't know":
        return "REFUSED — model said the context does not support an answer"
    return "CONFIDENT — model answered even though the only context was the wrong section"


def run_wrong_chunk(
    client: Groq,
    query: str,
    document_id: str,
    cache: dict[tuple[str, str], list[dict]],
    prefer_heading: str | None = None,
    label: str | None = None,
) -> str:
    chunks = load_chunks(document_id)
    retrieved_headings = normal_headings_for(query, document_id, chunks, cache)
    wrong_chunk = pick_wrong_chunk(chunks, retrieved_headings, prefer_heading)
    result = generate_from_chunk(client, query, wrong_chunk)
    answer = result.get("answer", "")
    verdict = refusal_verdict(answer)

    title = label or "wrong-chunk test"
    print()
    print("=" * 72)
    print(f"WRONG CHUNK — {title}")
    print("=" * 72)
    print(f"query asked: {query}")
    print(f"document_id: {document_id}")
    print(f"section that was actually retrieved: {', '.join(retrieved_headings) or 'unknown'}")
    print(
        "wrong chunk used: "
        f"{wrong_chunk.get('heading')} | chunk {wrong_chunk.get('chunkNumber')} | "
        f"pages {format_pages(wrong_chunk)}"
    )
    print(f"wrong chunk text: {excerpt(wrong_chunk.get('chunk_text', ''), 400)}")
    print(f"verdict: {verdict}")
    print(f"model answer: {answer}")
    print(f"model citations: {result.get('citations', [])}")
    return verdict


def document_id_for_query(entries: list[dict], query: str) -> str:
    for entry in entries:
        if entry.get("query") == query and entry.get("document_id"):
            return entry["document_id"]
    raise KeyError(f"Query is not in the log, so no document_id was found: {query}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print query logs and test wrong-chunk answers.")
    parser.add_argument(
        "--logs-only",
        action="store_true",
        help="Print the query log and stop before the model calls.",
    )
    parser.add_argument(
        "--wrong-chunk",
        action="store_true",
        help="Run one wrong-chunk test. Requires --query and --document-id.",
    )
    parser.add_argument("--query", help="Question to ask against a deliberately wrong chunk.")
    parser.add_argument("--document-id", help="Document whose chunks supply the wrong context.")
    parser.add_argument(
        "--prefer-heading",
        help="Section heading to force in as the wrong chunk, when it is not a retrieved section.",
    )
    parser.add_argument(
        "--skip-suite",
        action="store_true",
        help="Do not run the three logged questions (Ta'if, Abu Talib, Abyssinia).",
    )
    return parser.parse_args()


def main() -> None:
    _configure_stdout()
    load_dotenv(ROOT / ".env")
    args = parse_args()
    entries = load_logs(LOG_FILE)
    cache = load_cache_sources(CHROMA_DB)

    if not args.wrong_chunk:
        print_logs(entries, cache)

    if args.logs_only:
        return

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is missing from .env; wrong-chunk tests were not run.")
    client = Groq(api_key=api_key)

    if args.wrong_chunk:
        if not args.query or not args.document_id:
            raise SystemExit("--wrong-chunk requires both --query and --document-id")
        run_wrong_chunk(
            client,
            args.query,
            args.document_id,
            cache,
            prefer_heading=args.prefer_heading,
        )
        return

    if args.skip_suite:
        return

    print()
    print("=" * 72)
    print("WRONG-CHUNK SUITE")
    print("Each call sends one chunk from a section that was not retrieved for that question.")
    print("=" * 72)
    verdicts = []
    for case in SUITE:
        document_id = document_id_for_query(entries, case["query"])
        verdict = run_wrong_chunk(
            client,
            case["query"],
            document_id,
            cache,
            prefer_heading=case["prefer_heading"],
            label=case["label"],
        )
        verdicts.append((case["label"], verdict))

    print()
    print("=" * 72)
    print("SUITE SUMMARY")
    print("=" * 72)
    for label, verdict in verdicts:
        print(f"{label}: {verdict}")


if __name__ == "__main__":
    main()
