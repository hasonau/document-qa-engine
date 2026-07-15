# document-qa-engine

Building a RAG (Retrieval-Augmented Generation) pipeline from scratch in raw Python, without frameworks like LangChain, to understand the core mechanics.

## Progress

Day 1–6 complete: PDF extraction, chunking, embeddings (sentence-transformers), similarity search (sklearn → FAISS), and a full retrieval pipeline.

## Tech stack

Python, pdfplumber, sentence-transformers, FAISS, NumPy

## Layout

```
data/building/          shared sample PDF
learning/               study notes
1.Extraction/           PDF text extraction
2.Chunking/             text chunking
3.Embeddings/           sentence embeddings
4.Similarity_Search/    cosine similarity (sklearn)
5.Faiss_Implementation/ FAISS vector search
6.Retrieval Pipeline/   end-to-end retrieval
```

Each numbered folder holds that day's notebook only. Shared assets live at the repo root under `data/` and `learning/`.

## Note

The test document is a short PDF about Muhammad's biography, used only as sample text for building the pipeline.
