from fastapi import APIRouter, File, UploadFile, Cookie, Response, Request, HTTPException
from pydantic import BaseModel
import uuid
from month1_rag_engine import extract_pages, detect_sections, chunk_sections
from groq import Groq
import os
import logging
from ..services.rag import (
    answer_question,
    create_chromadb_params,
    save_to_chromadb,
    get_collection,
)
from ..services.security import (
    InMemoryRateLimiter,
    is_llm_upstream_failure,
    is_pdf_magic,
    sanitize_filename,
    validate_query_text,
)
from .. import config as app_config
from sse_starlette.sse import EventSourceResponse
import json
from rank_bm25 import BM25Okapi
import pickle

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path


logger = logging.getLogger(__name__)

LOG_FILE = Path("logs/query_logs.jsonl")
DOCUMENTS_DIR = Path("month2_api_service/documents")
BM25_DIR = Path(".")

rate_limiter = InMemoryRateLimiter()

# Test hook: set to a stage name to force failure during upload (integration tests).
# Values: "after_save" | "after_chroma" | None
_INGEST_FAIL_AFTER: str | None = os.getenv("INGEST_FAIL_AFTER") or None


def log_query(
    query: str,
    answer: str,
    document_id: str,
    served_from_cache: bool,
    success: bool,
):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "query": query,
        "answer": answer,
        "document_id": document_id,
        "timestamp": datetime.now(ZoneInfo("Asia/Karachi")).isoformat(),
        "served_from_cache": served_from_cache,
        "success": success,
    }

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def require_session_id(session_id: str | None) -> str:
    if session_id is None:
        raise HTTPException(status_code=401, detail="No session")
    return session_id


def parse_document_uuid(document_id: str) -> str:
    """Return canonical UUID string, or raise 404 for invalid ids."""
    try:
        return str(uuid.UUID(document_id))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(status_code=404, detail="Document not found")


def _path_inside_dir(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _pdf_info_for_document(document_id: str) -> tuple[str | None, str | None]:
    """Filename and uploaded_at ISO from disk PDF matching {document_id}_*."""
    docs_dir = DOCUMENTS_DIR
    if not docs_dir.is_dir():
        return None, None

    filename = None
    uploaded_at = None
    prefix = f"{document_id}_"
    for candidate in docs_dir.glob(f"{document_id}_*"):
        if not candidate.is_file():
            continue
        if not _path_inside_dir(candidate, docs_dir):
            continue
        name = candidate.name
        if not name.startswith(prefix):
            continue
        filename = name[len(prefix):] or None
        uploaded_at = datetime.fromtimestamp(
            candidate.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        break
    return filename, uploaded_at


def _bm25_path_for_document(document_id: str) -> Path:
    path = (BM25_DIR / f"bm25_{document_id}.pkl").resolve()
    if not _path_inside_dir(path, BM25_DIR):
        raise RuntimeError("BM25 path escaped expected directory")
    return path


def _delete_pdfs_for_document(document_id: str) -> int:
    """Delete PDFs matching {document_id}_* inside DOCUMENTS_DIR. Returns count removed."""
    docs_dir = DOCUMENTS_DIR
    if not docs_dir.is_dir():
        return 0

    removed = 0
    for candidate in list(docs_dir.glob(f"{document_id}_*")):
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if not _path_inside_dir(resolved, docs_dir):
            raise RuntimeError("PDF path escaped documents folder")
        resolved.unlink(missing_ok=True)
        removed += 1
    return removed


def _delete_chroma_chunks_for_document(document_id: str, session_id: str | None = None) -> int:
    """Delete Chroma document chunks for document_id (optionally scoped to session)."""
    collection = get_collection("documents")
    if session_id is not None:
        where = {
            "$and": [
                {"session_id": {"$eq": session_id}},
                {"document_id": {"$eq": document_id}},
            ]
        }
    else:
        where = {"document_id": {"$eq": document_id}}
    owned = collection.get(where=where, include=["metadatas"])
    owned_ids = owned.get("ids") or []
    if owned_ids:
        collection.delete(ids=owned_ids)
    return len(owned_ids)


def cleanup_upload_artifacts(document_id: str, session_id: str | None = None) -> dict:
    """Remove PDF, BM25, and Chroma chunks for a failed or rejected upload."""
    pdfs = _delete_pdfs_for_document(document_id)
    bm25_deleted = False
    try:
        bm25_path = _bm25_path_for_document(document_id)
        if bm25_path.is_file():
            bm25_path.unlink()
            bm25_deleted = True
    except Exception:
        logger.exception("cleanup: BM25 remove failed for document")
    chroma_deleted = 0
    try:
        chroma_deleted = _delete_chroma_chunks_for_document(document_id, session_id)
    except Exception:
        logger.exception("cleanup: Chroma remove failed for document")
    return {
        "pdfs": pdfs,
        "bm25": bm25_deleted,
        "chroma_chunks": chroma_deleted,
    }


def count_documents_for_session(session_id: str) -> int:
    collection = get_collection("documents")
    result = collection.get(
        where={"session_id": {"$eq": session_id}},
        include=["metadatas"],
    )
    ids: set[str] = set()
    for metadata in result.get("metadatas") or []:
        if not metadata:
            continue
        doc_id = metadata.get("document_id")
        if doc_id:
            ids.add(doc_id)
    return len(ids)


def _request_id(request: Request | None) -> str:
    if request is None:
        return "unknown"
    return getattr(request.state, "request_id", None) or "unknown"


def _rate_limit_or_429(key: str, limit: int) -> None:
    allowed, retry_after = rate_limiter.check(
        key,
        limit,
        float(app_config.RATE_LIMIT_WINDOW_SECONDS),
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again shortly.",
            headers={"Retry-After": str(retry_after)},
        )


async def _read_upload_capped(upload: UploadFile, max_bytes: int) -> bytes:
    """Read upload body; reject early with 413 if larger than max_bytes."""
    chunks: list[bytes] = []
    total = 0
    while True:
        piece = await upload.read(64 * 1024)
        if not piece:
            break
        total += len(piece)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File too large. Maximum upload size is "
                    f"{max_bytes // (1024 * 1024)} MB."
                ),
            )
        chunks.append(piece)
    return b"".join(chunks)


router = APIRouter()


class Query(BaseModel):
    query: str
    document_id: str


@router.get("/")
def read_root():
    return {"message": "Hello World"}


@router.get("/healthz")
def healthz():
    return {"status": "ok"}


@router.get("/documents")
def list_documents(request: Request):
    session_id = require_session_id(request.cookies.get("session_id"))

    collection = get_collection("documents")
    result = collection.get(
        where={"session_id": {"$eq": session_id}},
        include=["metadatas"],
    )

    counts: dict[str, int] = {}
    for metadata in result.get("metadatas") or []:
        if not metadata:
            continue
        doc_id = metadata.get("document_id")
        if not doc_id:
            continue
        counts[doc_id] = counts.get(doc_id, 0) + 1

    documents = []
    for document_id, chunk_count in sorted(counts.items()):
        filename, uploaded_at = _pdf_info_for_document(document_id)
        documents.append(
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_count": chunk_count,
                "uploaded_at": uploaded_at,
            }
        )

    return {"documents": documents}


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, request: Request):
    session_id = require_session_id(request.cookies.get("session_id"))
    document_id = parse_document_uuid(document_id)
    request_id = _request_id(request)

    documents_collection = get_collection("documents")
    owned = documents_collection.get(
        where={
            "$and": [
                {"session_id": {"$eq": session_id}},
                {"document_id": {"$eq": document_id}},
            ]
        },
        include=["metadatas"],
    )
    owned_ids = owned.get("ids") or []
    if not owned_ids:
        raise HTTPException(status_code=404, detail="Document not found")

    completed: list[str] = []
    failed_stage: str | None = None
    cache_deleted = 0
    bm25_deleted = False
    pdfs_deleted = 0
    chroma_deleted = 0

    # 1) Purge query cache (idempotent)
    try:
        cache_collection = get_collection("query_cache")
        cached = cache_collection.get(
            where={"document_id": {"$eq": document_id}},
            include=["metadatas"],
        )
        cache_ids = cached.get("ids") or []
        if cache_ids:
            cache_collection.delete(ids=cache_ids)
            cache_deleted = len(cache_ids)
        completed.append("cache")
    except Exception:
        failed_stage = "cache"
        logger.exception("delete cache failed request_id=%s", request_id)

    # 2) Delete BM25 pickle (idempotent)
    if failed_stage is None:
        try:
            bm25_path = _bm25_path_for_document(document_id)
            if bm25_path.is_file():
                bm25_path.unlink()
                bm25_deleted = True
            completed.append("bm25")
        except Exception:
            failed_stage = "bm25"
            logger.exception("delete bm25 failed request_id=%s", request_id)

    # 3) Delete uploaded PDF(s) (idempotent)
    if failed_stage is None:
        try:
            pdfs_deleted = _delete_pdfs_for_document(document_id)
            completed.append("pdf")
        except Exception:
            failed_stage = "pdf"
            logger.exception("delete pdf failed request_id=%s", request_id)

    # 4) Delete Chroma chunks LAST (only after earlier stages succeed)
    if failed_stage is None:
        try:
            documents_collection.delete(
                where={
                    "$and": [
                        {"session_id": {"$eq": session_id}},
                        {"document_id": {"$eq": document_id}},
                    ]
                }
            )
            chroma_deleted = len(owned_ids)
            completed.append("chroma")
        except Exception:
            failed_stage = "chroma"
            logger.exception("delete chroma failed request_id=%s", request_id)

    if failed_stage is not None:
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Document delete failed; retry safe for completed stages. "
                    f"Request id: {request_id}"
                ),
                "document_id": document_id,
                "completed": completed,
                "failed": failed_stage,
                "deleted": {
                    "cache_entries": cache_deleted,
                    "bm25": bm25_deleted,
                    "pdfs": pdfs_deleted,
                    "chroma_chunks": chroma_deleted,
                },
            },
        )

    return {
        "document_id": document_id,
        "deleted": {
            "cache_entries": cache_deleted,
            "bm25": bm25_deleted,
            "pdfs": pdfs_deleted,
            "chroma_chunks": chroma_deleted,
        },
        "completed": completed,
    }


@router.post("/query")
def ask_question(request: Request, query: Query):
    session_id = request.cookies.get("session_id")
    if session_id is None:
        raise HTTPException(status_code=401, detail="No session")

    request_id = _request_id(request)

    problem = validate_query_text(query.query, app_config.MAX_QUERY_CHARS)
    if problem == "empty":
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )
    if problem == "too_long":
        raise HTTPException(
            status_code=422,
            detail=(
                f"Question is too long. Maximum length is "
                f"{app_config.MAX_QUERY_CHARS} characters."
            ),
        )

    _rate_limit_or_429(
        f"query:{session_id}",
        app_config.QUERY_RATE_LIMIT_PER_MINUTE,
    )

    served_from_cache = False
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        result = answer_question(
            query.query,
            query.document_id,
            client,
            session_id,
        )
        served_from_cache = result["from_cache"]

        if served_from_cache:
            answer = result["answer"]
            cached_metadatas = result["cached_metadatas"]

            def generate_cached():
                try:
                    sources = json.loads(cached_metadatas[0]["sources"])
                    log_query(
                        query.query, answer, query.document_id, served_from_cache, True
                    )
                    yield {"event": "answer", "data": answer}
                    yield {"event": "sources", "data": json.dumps(sources)}
                except Exception:
                    log_query(
                        query.query, "", query.document_id, served_from_cache, False
                    )
                    raise

            return EventSourceResponse(generate_cached())

        log_query(
            query.query, result["answer"], query.document_id, served_from_cache, True
        )

        return {
            "answer": result["answer"],
            "citations": result["citations"],
        }
    except HTTPException:
        raise
    except Exception as e:
        log_query(query.query, "", query.document_id, served_from_cache, False)
        if is_llm_upstream_failure(e):
            logger.exception(
                "LLM upstream failure request_id=%s", request_id
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "The language model is temporarily unavailable. Please try again."
                ),
            ) from None
        logger.exception("Query failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail=(
                f"An unexpected error occurred. Please try again. "
                f"Request id: {request_id}"
            ),
        ) from None


@router.post("/upload-document")
async def upload_document(
    request: Request,
    response: Response,
    document: UploadFile = File(...),
    session_id: str | None = Cookie(default=None),
):
    request_id = _request_id(request)

    if session_id is None:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id)

    _rate_limit_or_429(
        f"upload:{session_id}",
        app_config.UPLOAD_RATE_LIMIT_PER_MINUTE,
    )

    # Enforce session document cap before allocating a new id / writing bytes.
    try:
        existing = count_documents_for_session(session_id)
    except Exception:
        logger.exception("session doc count failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail=(
                f"An unexpected error occurred. Please try again. "
                f"Request id: {request_id}"
            ),
        ) from None

    if existing >= app_config.MAX_DOCS_PER_SESSION:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Document limit reached. This session may have at most "
                f"{app_config.MAX_DOCS_PER_SESSION} documents. "
                f"Delete one before uploading another."
            ),
        )

    raw = await _read_upload_capped(document, app_config.MAX_UPLOAD_BYTES)

    if not is_pdf_magic(raw):
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are supported.",
        )

    safe_name = sanitize_filename(
        document.filename, max_len=app_config.SAFE_FILENAME_MAX_LEN
    )
    document_id = str(uuid.uuid4())
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DOCUMENTS_DIR / f"{document_id}_{safe_name}"
    if not _path_inside_dir(filepath, DOCUMENTS_DIR):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    try:
        filepath.write_bytes(raw)

        fail_after = _INGEST_FAIL_AFTER or os.getenv("INGEST_FAIL_AFTER")
        if fail_after == "after_save":
            raise RuntimeError("forced ingest failure after_save")

        dictionary_for_pages = extract_pages(pdf_path=str(filepath))

        page_count = len(dictionary_for_pages)
        if page_count > app_config.MAX_PDF_PAGES:
            cleanup_upload_artifacts(document_id, session_id)
            raise HTTPException(
                status_code=422,
                detail=(
                    f"PDF has too many pages ({page_count}). "
                    f"Maximum is {app_config.MAX_PDF_PAGES}."
                ),
            )

        total_chars = sum(len(page.get("text") or "") for page in dictionary_for_pages)
        if total_chars < app_config.MIN_EXTRACTABLE_CHARS:
            cleanup_upload_artifacts(document_id, session_id)
            raise HTTPException(
                status_code=422,
                detail="No extractable text (scanned PDFs are not supported)",
            )

        sections, fallback = detect_sections(dictionary_for_pages)

        if not fallback:
            whole_document_text = " ".join(
                page["text"] or "" for page in dictionary_for_pages
            )
            fake_section = {
                "sectionNumber": 1,
                "heading": "Full Document",
                "startPage": dictionary_for_pages[0]["pageNo"],
                "endPage": dictionary_for_pages[-1]["pageNo"],
                "section_text": whole_document_text,
            }
            chunks = chunk_sections([fake_section], document_id)
        else:
            chunks = chunk_sections(sections, document_id)

        if not chunks:
            cleanup_upload_artifacts(document_id, session_id)
            raise HTTPException(
                status_code=422,
                detail="No extractable text (scanned PDFs are not supported)",
            )

        sparse_chunks = []
        for chunk in chunks:
            chunk["document_id"] = f"{document_id}"
            chunk["session_id"] = f"{session_id}"
            sparse_chunks.append(chunk["chunk_text"].split())

        ids, chunksText, metadata, embeddings = create_chromadb_params(chunks)
        save_to_chromadb(ids, embeddings, chunksText, metadata)

        if fail_after == "after_chroma":
            raise RuntimeError("forced ingest failure after_chroma")

        bm25 = BM25Okapi(sparse_chunks)
        for id_, chunk in zip(ids, chunks):
            chunk["id"] = id_

        bm25_path = _bm25_path_for_document(document_id)
        with open(bm25_path, "wb") as f:
            bm25_chunksObject = {"bm25": bm25, "chunks": chunks}
            pickle.dump(bm25_chunksObject, f)

    except HTTPException:
        raise
    except Exception:
        cleanup_upload_artifacts(document_id, session_id)
        logger.exception("Upload/ingest failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail=(
                f"An unexpected error occurred. Please try again. "
                f"Request id: {request_id}"
            ),
        ) from None

    return {
        "message": "Document uploaded and processed successfully",
        "document_id": document_id,
        "filename": safe_name,
        "chunks_count": len(chunks),
    }
