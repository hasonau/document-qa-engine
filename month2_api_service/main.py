from fastapi import FastAPI
from pydantic import BaseModel

from month1_rag_engine import ask, build_index, chunk_pages, extract_pages

class Query(BaseModel):
    query: str

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/query")
def ask_question(query: Query):
    return {"message": "Query Asked And Answered Successfully", "query": query}
