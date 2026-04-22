# CSE-GO Pipeline

AI-powered UPSC Civil Services Exam question practice platform.

## What it does

Builds a RAG (Retrieval Augmented Generation) pipeline that:
1. Downloads NCERT textbooks (Classes 6–12)
2. Extracts, chunks, and embeds text into a local vector database (ChromaDB)
3. Generates grounded UPSC-style MCQ questions using the Claude API
4. Adversarially verifies each question for factual accuracy
5. Parses and ingests 1197 real UPSC Prelims PYQs (2011–2025) for direct practice

## Setup

```bash
pip install pymupdf chromadb sentence-transformers anthropic requests
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Pipeline steps

```bash
# Step 1 — Download NCERT PDFs into ./source-docs/
python download_ncert.py

# Step 2 — Extract, chunk, embed → ChromaDB (collection: cse_knowledge_base)
python ingest.py

# Step 3 — Generate UPSC MCQs from retrieved chunks
python generate.py
# or with custom topics:
python generate.py "fundamental rights" "monsoon rainfall"

# Step 4 — Adversarial verification of generated questions
python check.py

# Step 5 — Parse real UPSC PYQs (2011–2025) from PDF
# Place "Prelims PYQ (2011-2025).pdf" in ./source-docs/ first
python parse_pyq.py

# Step 6 — Ingest PYQs into ChromaDB (collection: pyq_questions)
python ingest_pyq.py
```

## Collections in ChromaDB

| Collection | Contents |
|---|---|
| `cse_knowledge_base` | NCERT textbook chunks (Classes 6–12) |
| `pyq_questions` | 1197 real UPSC Prelims questions (2011–2025) with answers and explanations |

## Models used

- Embeddings: `all-MiniLM-L6-v2` (local, free, no API key)
- Generation: `claude-sonnet-4-6` (Anthropic API)
