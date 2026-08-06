from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel
import uuid
from month1_rag_engine import chunk_pages, extract_pages
from groq import Groq
import os
from ..services.rag import ask, create_chromadb_params, query_chromadb, save_to_chromadb

router = APIRouter()


class Query(BaseModel):
    query: str 
    document_id : str


@router.get("/")
def read_root():
    return {"message": "Hello World"}

@router.post("/query")
def ask_question(query: Query):

    result = query_chromadb(
        query.query,
        query.document_id
    )
    # GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )
    answer,found = ask(query.query,result,client)
    # no chunks came back so,no need to attach sources either
    response =  {
        "answer": answer,
        "sources": result["metadatas"][0]
        }
    return response if found else {"answer":answer,"sources":[]}

@router.post("/upload-document")
async def upload_document(document: UploadFile = File(...)):
    document_id = str(uuid.uuid4())
    # create folde first
    os.makedirs("month2_api_service/documents", exist_ok=True)  # create if not there,otherwise ignore
    
    filepath = f"month2_api_service/documents/{document_id}_{document.filename}"
    
    # write or save in that documents folder
    with open(filepath, "wb") as f:
        f.write(await document.read())
    
    dictionary_for_pages = extract_pages(pdf_path = filepath)
    # step 2
    # Chunk the pages into chunks
    chunks = chunk_pages(dictionary_for_pages)
    # add document id to each chunk
    for chunk in chunks:
        chunk["document_id"] = f"{document_id}"  

    # chromdadb params made
    ids, chunksText, metadata, embeddings = create_chromadb_params(chunks)
    # embeddings of chunks
    chromadb_collection = save_to_chromadb(ids,embeddings,chunksText,metadata) 
    

    return {
    "message": "Document uploaded and processed successfully",
    "document_id": document_id,
    "filename": document.filename,
    "chunks_count": len(chunks)
    }
