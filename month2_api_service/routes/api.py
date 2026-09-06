from fastapi import APIRouter, File, UploadFile ,Cookie, Response,Request,HTTPException
from pydantic import BaseModel
import uuid
from month1_rag_engine import extract_pages,detect_sections,chunk_sections
from groq import Groq
import os
from ..services.rag import ask, create_chromadb_params, query_chromadb, save_to_chromadb, query_sparse, rrf,rerank
from sse_starlette.sse import EventSourceResponse
import json
from rank_bm25 import BM25Okapi
import pickle

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

@router.post("/query")
def ask_question(request:Request,query: Query):
    
    session_id = request.cookies.get("session_id")
    if session_id is None:
        raise HTTPException(status_code=401, detail="No session")

    # dense results
    result = query_chromadb(
        query.query,
        query.document_id,
        session_id
    )

    query_tokens = query.query.split()
    # sparse results
    result_sparse = query_sparse(query_tokens, query.document_id)
    # rrf step
    fused = rrf(result,result_sparse)

    # for reranking

    chunks = {}

    for i, item_id in enumerate(result["ids"][0]):
        metadata = result["metadatas"][0][i]

        chunks[item_id] = {
            "chunk_text": result["documents"][0][i],
            "startPage": metadata["startPage"],
            "endPage": metadata["endPage"],
            "chunkNumber": metadata["chunkNumber"],
            "sectionNumber": metadata["sectionNumber"],
            "heading": metadata["heading"]
        }

    for item in result_sparse:
        chunks[item["id"]] = item

    # make pairs
    pairs = []
    candidate_ids =[]
    for item_id, score in fused:
        chunk_text = chunks[item_id]["chunk_text"]

        candidate_ids.append(item_id)
        pairs.append((query.query, chunk_text))

    rerank_scores = rerank(pairs)

    re_paired = list(zip(rerank_scores, pairs,candidate_ids)) 
    top_3 = sorted(re_paired, key=lambda x:x[0], reverse=True)
    top_3 = top_3[:3]

    reranked_fused = []

    for score, pair, candidate_id in top_3:
        reranked_fused.append({
            "id": candidate_id,
            "score": score,
            "chunk": chunks[candidate_id]
        })
    
    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )
    def generate():
        for label,value in ask(query.query, reranked_fused, client):
            if label == "not_found":
                yield{"event":"not_found", "data" : "Not in Documents"}
                return
            
            yield {"event": "answer", "data": value}
        chunks_results = [rerank_fuse["chunk"] for rerank_fuse in reranked_fused]
        yield {"event": "sources", "data": json.dumps(chunks_results)} 
        
    return EventSourceResponse(generate())

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
