import chromadb
from sentence_transformers import SentenceTransformer as ST

model = ST("all-MiniLM-L6-v2")


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
        yield ("not_found", None)
        return
    
    
    # concatenate all three things into one
    message = "Instructions :\n" + instructions + "\n Context :" + currentContext + "\n Question : \n" +query
    messages = [{"role": "user", "content": message}]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        stream=True
    )

    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield ("answer",content)


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
    return result


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
