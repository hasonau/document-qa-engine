from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import uuid
import chromadb


from month1_rag_engine import ask, build_index, chunk_pages, extract_pages
from sentence_transformers import SentenceTransformer as ST
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

class Query(BaseModel):
    query: str

model = ST("all-MiniLM-L6-v2")
app = FastAPI()


def chromadb_client_setup():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/query")
def ask_question(query: Query):
    # step 1
    # Extract the pages from the PDF
    dictionary_for_pages = extract_pages(pdf_path="month1_rag_engine/data/building/Muhammad-pages.pdf")
    # step 2
    # Chunk the pages into chunks
    chunks = chunk_pages(dictionary_for_pages)
    # step 3
    # Build the index
    index = build_index(chunks,model=model)
    # step 4
    # Initialize the client
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    # step 5 Reterieve the indices of the chunks
    query_embeddings = model.encode([query.query])
    distances, indices = index.search(query_embeddings, 3)
    # step 5
    # Ask the question
    answer = ask(query.query, chunks, indices, client)
    return {"message": "Query Asked And Answered Successfully", "query": query.query, "answer": answer}

@app.post("/upload-document")
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

    # embeddings of chunks
    chunksText=[]
    ids = []
    metadata = []
    for chunk in chunks:
        chunksText.append(chunk["chunk_text"])
        ids.append(f"{chunk["document_id"]}_{chunk['chunkNumber']}")
        metadata.append(
            {
                "document_id" : chunk["document_id"],
                "startPage" : chunk["startPage"],
                "chunkNumber" : chunk["chunkNumber"],
                "endPage" : chunk["endPage"]
            })

    embeddings = model.encode(chunksText)


    chromadb_client = chromadb_client_setup() 
    collection = chromadb_client.get_or_create_collection(name="documents")
    collection.add(ids=ids,embeddings = embeddings,documents = chunksText,metadatas= metadata)

    return {
    "message": "Document uploaded and processed successfully",
    "document_id": document_id,
    "filename": document.filename,
    "chunks_count": len(chunks)
    }