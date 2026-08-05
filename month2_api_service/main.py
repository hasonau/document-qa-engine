from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import uuid
import chromadb
from chromadb import K 

from month1_rag_engine import ask, build_index, chunk_pages, extract_pages
from sentence_transformers import SentenceTransformer as ST
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()



class Query(BaseModel):
    query: str 
    document_id : str

model = ST("all-MiniLM-L6-v2")
app = FastAPI()


def ask(query, result, client):

    instructions = ("\nAnswer the question using only the provided context. "
        "If the context does not contain enough information, respond exactly with 'I don't know'. "
        "If the answer is found, cite the source number, page number, and chunk number used.")

    sourceCount = 1
    # contexts = []
    currentContext = ""
    found = False
    for doc, metadata,distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        if distance <= 1.5 :
            found = True

            currentContext += f"Source {sourceCount} :\n"
            currentContext += f"\nPage Number: {metadata['startPage']}"
            if metadata["startPage"] != metadata["endPage"]:
                currentContext += f" - {metadata['endPage']}"
            currentContext += f"\nChunk Number: {metadata['chunkNumber']}\n"
            currentContext += doc + "\n"
            sourceCount+=1
    
    if not found:
        return "Not in documents",found
    
    
    # concatenate all three things into one
    message = "Instructions :\n" + instructions + "\n Context :" + currentContext + "\n Question : \n" +query
    messages = [{"role": "user", "content": message}]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )

    return response.choices[0].message.content,found


def create_chromadb_params(chunks):
    ids = []
    metadata = []
    chunksText = []

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
    return ids,chunksText,metadata,embeddings

def save_to_chromadb(ids, embeddings, documents, metadatas):
    collection = get_collection()
    collection.add(ids=ids,embeddings = embeddings,documents = documents,metadatas= metadatas)

def get_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection("documents")

def query_chromadb(question, document_id):

    query_embedding = model.encode([question])
    collection = get_collection()
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        where={
            "document_id": document_id  
        }
    )
    print(result)
    return result


@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.post("/query")
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