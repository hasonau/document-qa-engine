# LEARNING RELATED

## GPT PLAN MONTH 1:

# Core Idea (don’t skip this)

You are building 5 real systems:

1. Document loader (PDF → text)
2. Chunking engine
3. Embedding generator
4. Vector database (store + search)
5. Retriever + LLM answer generator

Everything in RAG = just these 5 parts.

---

# Week 1 — Foundations (No “RAG apps” yet)

## Day 1 — What RAG actually is (mental model)

**Watch topics:**

- “What is RAG (Retrieval Augmented Generation)”
- “Embeddings explained simply”
- “Vector similarity search basics”

**Build:**

- Take a PDF → extract raw text in Python
- Print chunks of raw text

Goal: just understand “data exists as text”

---

## Day 2 — Chunking (MOST IMPORTANT PART)

**Watch topics:**

- “Chunking strategies in RAG”
- recursive chunking vs fixed chunking
- overlap in text splitting

**Build:**

- split your PDF text into chunks:
    - 300–500 tokens/words
    - overlap 50–100
- print chunks + check readability

Goal: learn “bad chunking = bad AI”

---

## Day 3 — Embeddings (core brain of RAG)

**Watch topics:**

- “What are embeddings (intuitive)”
- “sentence transformers / OpenAI embeddings explained”

**Build:**

- convert each chunk → vector (embedding)
- store in array/list (NO DB yet)

Goal: understand “text becomes numbers”

---

## Day 4 — Similarity search (NO vector DB yet)

**Watch topics:**

- cosine similarity explained
- nearest neighbor search intuition

**Build:**

- write function:
    - query → embedding
    - compare with all chunk embeddings
    - return top 3 chunks

Goal: your first “search engine”

---

## Day 5 — Vector DB concept (theory + replacement)

**Watch topics:**

- “FAISS explained”
- “What is vector database”

**Build:**

- replace manual search with FAISS (or simple numpy still ok)
- store embeddings properly

Goal: you now understand real retrieval layer

---

## Day 6 — First full pipeline (NO LLM yet)

**Build:**

- PDF → chunks → embeddings → similarity search → return chunks

Input: question

Output: relevant text chunks

Goal: retrieval system ONLY

---

## Day 7 — Review + rebuild from scratch

**Task:**

- delete code mentally
- rebuild pipeline without looking

Goal: memory + clarity

---

# Week 2 — Full RAG System (real AI app)

## Day 8 — Add LLM (generation step)

**Watch topics:**

- “How RAG works end to end”
- “prompting with context”

**Build:**

- take top chunks
- send to LLM with prompt:
    - “Answer using only context below”

Goal: first real RAG system

---

## Day 9 — Prompt engineering for RAG

**Watch topics:**

- “RAG prompt design best practices”

**Build:**

- improve prompt:
    - context formatting
    - “if not found say I don’t know”

Goal: reduce hallucination

---

## Day 10 — Metadata (basic enterprise thinking)

**Watch topics:**

- “metadata filtering in RAG”

**Build:**

- attach metadata:
    - page number
    - section title

Goal: retrieval becomes structured

---

## Day 11 — Better chunking (hierarchical thinking)

**Watch topics:**

- recursive chunking
- structure-aware chunking

**Build:**

- chunk by headings if possible
- compare results with Day 2 chunks

Goal: understand why chunking matters deeply

---

## Day 12 — Query improvement (important but ignored by beginners)

**Watch topics:**

- query expansion
- HyDE (just concept)

**Build:**

- rewrite user query slightly before search
- try improved retrieval

Goal: better search results

---

## Day 13 — Evaluation (very important for jobs)

**Watch topics:**

- “RAG evaluation basics”
- precision vs recall in retrieval

**Build:**

- test 10 questions:
    - check correct chunk retrieval rate

Goal: understand system quality

---

## Day 14 — Final project polish

**Build:**

- clean pipeline
- CLI or simple web API (optional MERN backend fits here)
- document:
    - how it works
    - limitations

Goal: portfolio-ready RAG system

---

# What you will NOT do (important)

Avoid these for now:

- reranking models
- multi-hop agents
- graph RAG
- fine-tuning models
- LangChain dependency (you can study later)

---

# Final Truth (important mindset)

If you finish this plan properly, you will:

- understand 80% of real RAG systems used in companies
- be able to build production-level improvements later
- be ready for “AI backend engineer” roles

This is already closer to Google-style thinking than framework tutorials.

## GPT PLAN FOR MONTH 2:

Here’s **Month 2 = “production-level RAG + MERN integration + job-ready system”**. Goal: move from “I built RAG” → “I can ship an AI backend product”.

You’ll now focus on: **scalability, structure, APIs, and real app behavior**.

---

# Month 2 Goal (very clear)

By end of 30 days you should have:

- MERN app (frontend + backend)
- RAG backend as a service (API-based)
- multiple PDFs / documents support
- persistent vector DB (not in-memory)
- proper retrieval pipeline (production style)
- deployable project (portfolio-level)

---

# Project You Will Build (single system)

### “AI Document Assistant”

Example:

- upload PDFs (notes, books, docs)
- ask questions in chat UI
- get grounded answers from your docs

Think: **mini “ChatGPT for your files”**

---

# Tech Stack (fixed for Month 2)

### Backend (RAG service)

- Node.js / Express (or keep Python if you want simplicity, but MERN suggests Node API wrapper)
- Python RAG engine (optional microservice OR direct Node + vector DB)

### Vector DB

- FAISS (local) OR
- Chroma (recommended for ease) OR
- Qdrant (production feel)

### Frontend

- React
- simple chat UI + upload UI

---

# Week 1 — Turn RAG into a service (API layer)

## Day 1 — Convert RAG into API

**Build:**

- `/upload-document`
- `/query`

RAG becomes a backend service.

You stop thinking “script”, start thinking “system”.

---

## Day 2 — Document ingestion pipeline

**Build:**

- upload PDF → store file → extract text → chunk → embed → store vectors

Add structure:

- document_id
- chunk_id
- metadata

---

## Day 3 — Persistent storage (no more memory-only)

**Build:**

- save embeddings in DB (FAISS/Chroma/Qdrant)
- reload them on server restart

Goal: production requirement #1 solved

---

## Day 4 — Multi-document system

**Build:**

- multiple PDFs per user
- filter retrieval by document_id

This is where most beginners fail.

---

## Day 5 — Better retrieval pipeline

**Add:**

- top-k tuning
- metadata filtering
- score thresholding (ignore bad matches)

Goal: clean retrieval, not just “top 3 chunks”

---

## Day 6 — Response quality layer

**Build:**

- context formatting improvement
- “if not found → say not in documents”
- structured prompt template

Goal: production-grade prompting

---

## Day 7 — Mini review + refactor

- clean code structure:
    - /services/rag
    - /routes/api
    - /utils/embeddings

---

# Week 2 — MERN integration (real app)

## Day 8 — React UI setup

**Build:**

- chat UI
- message history
- simple layout

---

## Day 9 — File upload UI

**Build:**

- upload PDF from frontend
- send to backend API

---

## Day 10 — Connect chat → RAG backend

**Build:**

- user question → API → response → UI render

Now your system feels like ChatGPT clone

---

## Day 11 — Streaming responses (important for real apps)

**Learn:**

- streaming responses (SSE or websocket basics)

**Build:**

- response appears word-by-word

---

## Day 12 — User session system (basic)

**Build:**

- each user has separate document space
- session-based memory (simple)

---

## Day 13 — Improve UX + reliability

**Add:**

- loading states
- error handling
- retry logic

---

## Day 14 — Deployment (VERY IMPORTANT)

**Deploy:**

- backend → Render / Railway
- frontend → Vercel
- vector DB → local or hosted (if Qdrant cloud)

---

# Week 3 — “Production upgrades” (this is where you stand out)

## Day 15 — Hybrid search (optional but powerful)

- keyword + vector search

## Day 16 — Better chunking strategy v2

- structure-aware chunking (headings, sections)

## Day 17 — Query rewriting

- improve user question before search

## Day 18 — Basic caching

- store repeated queries

## Day 19 — Logging system

- log queries + results

## Day 20 — Debug retrieval failures

- analyze wrong answers

## Day 21 — Refactor + stabilize system

---

# Week 4 — Portfolio level polishing

## Day 22 — Add “knowledge base UI”

- show uploaded documents
- delete / re-upload

## Day 23 — Add “source citations”

- show which chunk was used

## Day 24 — Improve answer formatting

- structured answers (bullet + explanation)

## Day 25 — Edge case handling

- empty docs
- irrelevant questions

## Day 26 — Performance optimization

- reduce embedding calls
- batch processing

## Day 27 — Security basics

- limit file size
- sanitize inputs

## Day 28 — Final polish UI

## Day 29 — Write project documentation

- architecture diagram
- system explanation
- tech decisions

## Day 30 — Final deployment + GitHub portfolio

---

# What recruiters will see (important)

If done properly, your project shows:

- backend engineering skills
- AI system understanding (not just API usage)
- database + search systems
- full-stack capability
- production thinking

This is already **job-level AI engineer portfolio**, not beginner RAG.

---

# Reality check (important)

By end of Month 2:

You will NOT be:

- ML researcher
- transformer expert

You WILL be:

- AI backend engineer (RAG systems)
- capable of building SaaS AI tools
- ready for remote roles / startups

---

If you want next step after this:

I can design your **Month 3: “advanced RAG engineer (reranking, agents, eval frameworks, multi-hop systems like enterprise tools)”** which is where Google-level interviews start touching.