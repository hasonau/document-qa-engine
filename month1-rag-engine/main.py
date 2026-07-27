"""
CLI entry point for the document-qa-engine RAG pipeline.
Logic mirrors the day-folder notebooks (esp. 11.Hierarchical_Chunking / 12.Query_Rewriting).
"""

import argparse
import os
import re
import warnings

import faiss
import pdfplumber as pdf
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer as ST

warnings.filterwarnings("ignore")


def extract_pages(pdf_path):
    dictionary_for_pages = []
    with pdf.open(pdf_path) as pdfFile:
        for pageNo, content in enumerate(pdfFile.pages):
            dictionary_for_pages.append({"pageNo": pageNo, "text": content.extract_text()})
    return dictionary_for_pages


def chunk_pages(dictionary_for_pages):
    chunks = []
    chunkNumber = 1

    heading = "Default Heading"
    current_text = []
    startPage = None
    endPage = None

    for singleDictionary in dictionary_for_pages:
        pageNo = singleDictionary["pageNo"]
        pageText = singleDictionary["text"] or ""

        lines = pageText.split("\n")

        for line in lines:
            line = line.strip()

            if not line:
                continue

            cleaned_line = re.sub(r"\[\d+\]", "", line)
            words = cleaned_line.split()

            # heading detected
            if len(words) < 8 and not cleaned_line.endswith("."):

                if current_text:
                    chunks.append({
                        "startPage": startPage,
                        "endPage": endPage,
                        "chunkNumber": chunkNumber,
                        "heading": heading,
                        "chunk_text": " ".join(current_text).strip(),
                    })

                    chunkNumber += 1

                heading = line
                current_text = []

                # new chunk starts from current page
                startPage = pageNo

            else:
                # first line of chunk
                if startPage is None:
                    startPage = pageNo

                # update every time text is added
                endPage = pageNo

                current_text.append(line)

    # push last remaining chunk
    if current_text:
        chunks.append({
            "startPage": startPage,
            "endPage": endPage,
            "chunkNumber": chunkNumber,
            "heading": heading,
            "chunk_text": " ".join(current_text).strip(),
        })

    return chunks


def build_index(chunks, model):
    chunksText = []
    for chunk in chunks:
        chunksText.append(chunk["chunk_text"])
    embeddings = model.encode(chunksText)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def ask(query, chunks, indices, client):
    message = (
        query + "\n Answer the above question,only using the text below ,and "
        "say I don't know if not found ,and also if you do find an answer then also tell, "
        "which page number and chunk number u used,if startpage and end page are different ,then tell both,otherwise only one "
    )

    for i in indices[0]:
        message += f"Page Number = {chunks[i]['startPage']}\n"
        message += f"Page Number = {chunks[i]['endPage']}\n"
        message += f"Chunk Number = {chunks[i]['chunkNumber']}\n"
        message += chunks[i]["chunk_text"] + "\n"

    messages = [{"role": "user", "content": message}]
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )
    return response


def main():
    parser = argparse.ArgumentParser(
        description="Ask a question over a PDF using the from-scratch RAG pipeline."
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("question", help="Question to ask about the document")
    args = parser.parse_args()

    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY not found. Set it in a .env file at the repo root.")

    client = Groq(api_key=GROQ_API_KEY)

    print("Extracting text...")
    dictionary_for_pages = extract_pages(args.pdf)

    print("Chunking...")
    chunks = chunk_pages(dictionary_for_pages)
    if not chunks:
        raise SystemExit("No chunks produced from the PDF.")

    print("Embedding and indexing...")
    model = ST("all-MiniLM-L6-v2")
    index = build_index(chunks, model)

    print("Retrieving...")
    query_embeddings = model.encode([args.question])
    distances, indices = index.search(query_embeddings, 3)

    print("Generating answer...\n")
    response = ask(args.question, chunks, indices, client)
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
