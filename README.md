# document-qa-engine
Building a RAG (Retrieval-Augmented Generation) pipeline from scratch in raw Python, without frameworks like LangChain, to understand the core mechanics.
## Progress
Day 1–11 complete: PDF extraction, chunking, embeddings (sentence-transformers), similarity search (sklearn → FAISS), full retrieval pipeline, from-memory rebuild, LLM generation (Groq), prompt engineering with hallucination guardrails + self-reported context citation, page/chunk metadata tracking, and heading-based hierarchical chunking with cross-page start/end page tracking.
## Tech stack
Python, pdfplumber, sentence-transformers, FAISS, NumPy, Groq (llama-3.3-70b-versatile)
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
7.Rebuild/              from-memory pipeline rebuild
8.LLM_Generation/       RAG loop with Groq LLM
9.Prompt_Engineering/   context labeling, hallucination guardrails, citation
10.Metadata/            page number and chunk id metadata
11.Hierarchical_Chunking/ heading-based chunking with startPage/endPage tracking
```
Each numbered folder holds that day's notebook only. Shared assets live at the repo root under `data/` and `learning/`.
## Note
The test document is a short PDF about Muhammad's biography, used only as sample text for building the pipeline.