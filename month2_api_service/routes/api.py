from fastapi import APIRouter, File, UploadFile ,Cookie, Response,Request,HTTPException
from pydantic import BaseModel
import uuid
from month1_rag_engine import extract_pages,detect_sections,chunk_sections
from groq import Groq
import os
from ..services.rag import (
    answer_question,
    create_chromadb_params,
    save_to_chromadb,
    get_collection,
)
from sse_starlette.sse import EventSourceResponse
import json
from rank_bm25 import BM25Okapi
import pickle

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path



LOG_FILE = Path("logs/query_logs.jsonl")
DOCUMENTS_DIR = Path("month2_api_service/documents")
BM25_DIR = Path(".")


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


router = APIRouter()


class Query(BaseModel):
    query: str 
    document_id : str


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
    error_detail: str | None = None
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
    except Exception as e:
        failed_stage = "cache"
        error_detail = str(e)

    # 2) Delete BM25 pickle (idempotent)
    if failed_stage is None:
        try:
            bm25_path = _bm25_path_for_document(document_id)
            if bm25_path.is_file():
                bm25_path.unlink()
                bm25_deleted = True
            completed.append("bm25")
        except Exception as e:
            failed_stage = "bm25"
            error_detail = str(e)

    # 3) Delete uploaded PDF(s) (idempotent)
    if failed_stage is None:
        try:
            pdfs_deleted = _delete_pdfs_for_document(document_id)
            completed.append("pdf")
        except Exception as e:
            failed_stage = "pdf"
            error_detail = str(e)

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
        except Exception as e:
            failed_stage = "chroma"
            error_detail = str(e)

    if failed_stage is not None:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Document delete failed; retry safe for completed stages",
                "document_id": document_id,
                "completed": completed,
                "failed": failed_stage,
                "error": error_detail,
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
def ask_question(request:Request,query: Query):
    
    session_id = request.cookies.get("session_id")
    if session_id is None:
        raise HTTPException(status_code=401, detail="No session")

    served_from_cache = False
    try:

        client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
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
                    log_query(query.query, answer, query.document_id, served_from_cache, True)
                    yield {"event": "answer", "data": answer}
                    yield {"event": "sources", "data": json.dumps(sources)}
                except Exception:
                    log_query(query.query, "", query.document_id, served_from_cache, False)
                    raise

            return EventSourceResponse(generate_cached())

        log_query(query.query,result["answer"],query.document_id,served_from_cache,True)

        return {
                "answer": result["answer"],
                "citations": result["citations"]
            }
    except Exception as e:
        log_query(query.query,"",query.document_id,served_from_cache,False)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-document")
async def upload_document(response: Response,document: UploadFile = File(...),session_id: str | None = Cookie(default=None)):
    if session_id is None:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id)
    document_id = str(uuid.uuid4())
    # create folder first
    os.makedirs("month2_api_service/documents", exist_ok=True)  # create if not there,otherwise ignore
    
    filepath = f"month2_api_service/documents/{document_id}_{document.filename}"
    
    # write or save in that documents folder
    with open(filepath, "wb") as f:
        f.write(await document.read())
    
    # extract pages
    dictionary_for_pages = extract_pages(pdf_path=filepath)

    # detect sections / headings
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
            "section_text": whole_document_text
        }

        chunks = chunk_sections([fake_section], document_id)

    else:
        chunks = chunk_sections(sections, document_id)

    # make chunks
    sparse_chunks =[]
    # add document id to each chunk
    for chunk in chunks:
        chunk["document_id"] = f"{document_id}" 
        chunk["session_id"] = f"{session_id}" 
        sparse_chunks.append(chunk["chunk_text"].split())

    # chromdadb params made
    ids, chunksText, metadata, embeddings = create_chromadb_params(chunks)
    # dense embeddings of chunks
    chromadb_collection = save_to_chromadb(ids,embeddings,chunksText,metadata) 

    # add the ids to sparse_chunks as well
    # sparse index
    bm25 = BM25Okapi(sparse_chunks)
    # id attachement for RRF useage later
    for id,chunk in zip(ids,chunks):
        chunk["id"] = id
    # save to disk
    with open(f"bm25_{document_id}.pkl", "wb") as f:
        bm25_chunksObject = {}
        bm25_chunksObject["bm25"] = bm25
        bm25_chunksObject["chunks"] = chunks
        
        pickle.dump(bm25_chunksObject, f)

    

    return {
    "message": "Document uploaded and processed successfully",
    "document_id": document_id,
    "filename": document.filename,
    "chunks_count": len(chunks)
    }
