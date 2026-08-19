from fastapi import APIRouter, File, UploadFile ,Cookie, Response,Request,HTTPException
from pydantic import BaseModel
import uuid
from month1_rag_engine import chunk_pages, extract_pages
from groq import Groq
import os
from ..services.rag import ask, create_chromadb_params, query_chromadb, save_to_chromadb,query_sparse
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

    
    result = query_chromadb(
        query.query,
        query.document_id,
        session_id
    )

    query_tokens = query.query.split()

    result_sparse = query_sparse(query_tokens,query.document_id)

    # GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )
    def generate():
        for label,value in ask(query.query, result, client):
            if label == "not_found":
                yield{"event":"not_found", "data" : "Not in Documents"}
                return
            
            yield {"event": "answer", "data": value}
        yield {"event": "sources", "data": json.dumps(result["metadatas"][0])}     
        
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
    
    dictionary_for_pages = extract_pages(pdf_path = filepath)
    # step 2
    # Chunk the pages into chunks
    chunks = chunk_pages(dictionary_for_pages)
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

    # sparse index
    bm25 = BM25Okapi(sparse_chunks)
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
