from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from month1_rag_engine import ask, build_index, chunk_pages, extract_pages
from sentence_transformers import SentenceTransformer as ST
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

class Query(BaseModel):
    query: str

app = FastAPI()

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
    model = ST("all-MiniLM-L6-v2")
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
def upload_document(file: UploadFile = File(...)):
    return {"message": "Upload endpoint working"}