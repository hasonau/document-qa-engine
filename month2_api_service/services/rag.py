import chromadb
from sentence_transformers import SentenceTransformer as ST
from sentence_transformers import CrossEncoder
model = ST("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
import pickle
import numpy as np
import json 

def ask(query, fused, client):

    instructions = ("\nAnswer the question using only the provided context. "
        "If the context does not contain enough information, respond exactly with 'I don't know'. "
        "If the answer is found, cite the source number, page number, and chunk number used.")

    sourceCount = 1
    currentContext = ""
    found = True

    for fuse in fused:
        currentContext += f"Source {sourceCount} :\n"
        currentContext += f"\nSection: {fuse['chunk']['heading']}"
        currentContext += f"\nPage Number: {fuse['chunk']['startPage']}"

        if fuse["chunk"]["startPage"] != fuse["chunk"]["endPage"]:
            currentContext += f" - {fuse['chunk']['endPage']}"

        currentContext += f"\nChunk Number: {fuse['chunk']['chunkNumber']}\n"
        currentContext += fuse["chunk"]["chunk_text"] + "\n"

        sourceCount += 1
        
    
    if not found:
        yield ("not_found",found)
        return
    
    
    # concatenate all three things into one
    message = "Instructions :\n" + instructions + "\n Context :" + currentContext + "\n Question : \n" +query
    messages = [{"role": "user", "content": message}]

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
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


def query_chromadb(question, document_id,session_id):

    query_embedding = model.encode([question])
    collection = get_collection()
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=10,
        where={
            "$and": [
            {"session_id": {"$eq": session_id}},
            {"document_id": {"$eq" : document_id}},
        ]
        }
    )
    return result

def query_sparse(query_tokens,document_id):
    with open(f"bm25_{document_id}.pkl", "rb") as f:
        bm25_chunksObject = pickle.load(f)

        bm25 = bm25_chunksObject["bm25"]
        chunks = bm25_chunksObject["chunks"]
    scores = bm25.get_scores(query_tokens)
    top_k_indices = np.argsort(scores)[::-1][:10]
    top_k_chunks = [chunks[i] for i in top_k_indices]

    return top_k_chunks


def rrf(dense_results,sparse_chunks,k=60,top_n=10):

    scores = {}

    for index,item in enumerate(dense_results["ids"][0]):
        rank = index+1
        scores[item] = scores.get(item,0) + 1/(k+rank)

    for index,item in enumerate(sparse_chunks):
        rank = index+1
        id = item["id"]
        scores[id] = scores.get(id,0) + 1/(k+rank)
    
    scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return scores


def rerank(pairs):
    scores = reranker.predict(pairs)
    return scores

def query_expansion(user_query,client) -> list[str]:

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": """
                You are a query expansion model for a RAG retrieval system.

                Generate exactly 4 alternative search queries for the user's query.

                Rules:
                - Preserve the original intent.
                - Keep queries short and search-oriented.
                - Do not add unsupported assumptions.
                - If the query is ambiguous, do not guess a specific meaning.
                - Make the variants meaningfully different.
                - Return only the 4 queries; do not explain them.
                """
            },
            {
                "role": "user",
                "content": user_query,
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "query_expansion",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "expanded_queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 4,
                            "maxItems": 4
                        }
                    },
                    "required": ["expanded_queries"],
                    "additionalProperties": False
                }
            }
        }
    )

    result = json.loads(response.choices[0].message.content or "{}")
    return result["expanded_queries"]


def create_chromadb_params(chunks):
    ids = []
    metadata = []
    chunksText = []

    for chunk in chunks:
        chunksText.append(chunk["chunk_text"])
        ids.append(f"{chunk["document_id"]}_{chunk['sectionNumber']}_{chunk['chunkNumber']}")
        metadata.append(
            {
                 "document_id": chunk["document_id"],
                "session_id": chunk["session_id"],
                "sectionNumber": chunk["sectionNumber"],
                "heading": chunk["heading"],
                "startPage": chunk["startPage"],
                "endPage": chunk["endPage"],
                "chunkNumber": chunk["chunkNumber"]
            })

    embeddings = model.encode(chunksText)
    return ids,chunksText,metadata,embeddings

def save_to_chromadb(ids, embeddings, documents, metadatas):
    collection = get_collection()
    collection.add(ids=ids,embeddings = embeddings,documents = documents,metadatas= metadatas)
