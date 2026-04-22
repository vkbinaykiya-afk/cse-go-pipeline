# generate.py
# CSE-GO Pipeline — Step 2: Retrieve chunk → Generate MCQ via Claude API
#
# What this file does:
#   For each topic you provide, it finds the most relevant chunk from
#   ChromaDB, sends it to Claude with a strict "use only this chunk"
#   instruction, and saves the structured question as JSON.
#
# ── INSTALL (if not done already) ────────────────────────────────────────────
#
#   pip install anthropic chromadb sentence-transformers
#
# ── SET YOUR API KEY (run once in terminal, or add to your shell profile) ────
#
#   export ANTHROPIC_API_KEY="sk-ant-..."
#
# ── HOW TO RUN ────────────────────────────────────────────────────────────────
#
#   # Use the default built-in topics:
#   python generate.py
#
#   # Or pass your own topics as arguments:
#   python generate.py "fundamental rights" "monsoon rainfall" "parliamentary system"
#
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import datetime
import anthropic
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# ── CONFIGURATION ─────────────────────────────────────────────────────────────

CHROMA_DIR   = "./chroma-db"
COLLECTION   = "cse_knowledge_base"
EMBED_MODEL  = "all-MiniLM-L6-v2"
OUTPUT_DIR   = "./questions"
CLAUDE_MODEL = "claude-sonnet-4-6"      # change to claude-haiku-4-5 if you want cheaper/faster

# Top N chunks to retrieve per topic query.
# We use only the #1 result for generation — higher N helps us see alternatives.
TOP_K = 3

# Default topics if you run the script without arguments.
# These are common UPSC Prelims themes — feel free to edit.
DEFAULT_TOPICS = [
    "fundamental rights constitution India",
    "monsoon rainfall India geography",
    "parliamentary system India legislature",
    "photosynthesis plants biology",
    "economic development India GDP",
]

# ── PROMPT ────────────────────────────────────────────────────────────────────
# This is the instruction given to Claude for every question.
# The key constraint: "use ONLY the chunk below. No outside knowledge."

SYSTEM_PROMPT = """\
You are an expert UPSC Civil Services Examination (Prelims) question setter with 15 years of experience.

Your job is to generate ONE high-quality MCQ question for UPSC aspirants.

STRICT RULES — follow every one without exception:
1. Use ONLY the text chunk provided in the user message. Do not use any knowledge outside that chunk.
2. The correct answer must be directly and explicitly stated in the chunk. If it requires inference, do not generate the question.
3. Distractors (wrong options) must be plausible but clearly wrong based on the chunk. Do not invent false facts as distractors.
4. Do NOT write "According to the passage" or "As mentioned in the text". The question must read as a standalone factual question.
5. The cited_extract must be a verbatim copy of the sentence(s) from the chunk that prove the correct answer. No paraphrasing.
6. Output ONLY valid JSON — no explanation, no preamble, no markdown fences. Just the raw JSON object.\
"""

USER_PROMPT_TEMPLATE = """\
SOURCE CHUNK:
---
{chunk_text}
---

SOURCE METADATA:
  File  : {filename}
  Page  : {page}

Generate one UPSC-style MCQ question strictly from this chunk.

Return your response as a JSON object with exactly these keys:
{{
  "question":      "The question text (do not reference 'the passage')",
  "options": {{
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  }},
  "correct_answer": "A",
  "explanation":   "2-3 sentence explanation of why the correct answer is right.",
  "cited_extract": "Exact verbatim sentence(s) from the chunk that support the correct answer."
}}\
"""

# ─────────────────────────────────────────────────────────────────────────────


def load_collection():
    """
    Connect to the ChromaDB database that ingest.py created.
    Returns the collection object, or exits with a helpful message if not found.
    """
    if not os.path.exists(CHROMA_DIR):
        print(f"\nERROR: ChromaDB not found at '{CHROMA_DIR}'.")
        print("Run ingest.py first to build the knowledge base.")
        sys.exit(1)

    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn   = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embed_fn
    )

    count = collection.count()
    if count == 0:
        print(f"\nERROR: Collection '{COLLECTION}' is empty.")
        print("Run ingest.py first and make sure your PDFs are in source-docs/.")
        sys.exit(1)

    print(f"Connected to ChromaDB — {count} chunks available.")
    return collection


def retrieve_best_chunk(collection, topic):
    """
    Query ChromaDB with the topic string and return the single best matching chunk.

    ChromaDB converts the topic to an embedding using the same model as ingest.py,
    then finds the chunks whose embeddings are most similar (cosine similarity).

    Returns a dict:
        {
          "text":     "the chunk text",
          "source":   "filename.pdf",
          "page":     42,
          "chunk_index": 7,
          "distance": 0.23   (lower = more similar)
        }
    """
    results = collection.query(
        query_texts=[topic],
        n_results=TOP_K
    )

    # results is a dict of lists — index 0 is the best match
    best_text     = results["documents"][0][0]
    best_metadata = results["metadatas"][0][0]
    best_distance = results["distances"][0][0]

    return {
        "text":        best_text,
        "source":      best_metadata["source"],
        "page":        best_metadata["page"],
        "chunk_index": best_metadata["chunk_index"],
        "distance":    round(best_distance, 4)
    }


def generate_question(client, chunk):
    """
    Call the Claude API with the retrieved chunk and return a parsed question dict.

    Uses prompt caching on the system prompt — if you generate many questions
    in one run, subsequent calls reuse the cached system prompt and cost less.
    """

    user_message = USER_PROMPT_TEMPLATE.format(
        chunk_text=chunk["text"],
        filename=chunk["source"],
        page=chunk["page"]
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                # cache_control tells Claude to cache this system prompt
                # across multiple calls in this session — saves tokens + cost
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    raw_text = response.content[0].text.strip()

    # Claude was told to return only JSON — parse it directly
    # If parsing fails, we catch the error and return it as a flag
    try:
        question_data = json.loads(raw_text)
    except json.JSONDecodeError:
        # Sometimes Claude wraps JSON in markdown fences — strip them and retry
        cleaned = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            question_data = json.loads(cleaned)
        except json.JSONDecodeError:
            # If still broken, save the raw response so you can inspect it
            question_data = {"raw_response": raw_text, "parse_error": True}

    return question_data


def build_output_record(topic, chunk, question_data):
    """
    Combines the question with its source metadata into one complete record.
    This is what gets saved to disk and later reviewed by you.
    """
    return {
        "topic_query":   topic,
        "source": {
            "filename":    chunk["source"],
            "page":        chunk["page"],
            "chunk_index": chunk["chunk_index"],
            "similarity":  chunk["distance"]    # lower = better match
        },
        "question":      question_data.get("question"),
        "options":       question_data.get("options"),
        "correct_answer":question_data.get("correct_answer"),
        "explanation":   question_data.get("explanation"),
        "cited_extract": question_data.get("cited_extract"),
        "status":        "pending_check",   # check.py will update this to pass/flag
        "parse_error":   question_data.get("parse_error", False),
        "raw_response":  question_data.get("raw_response")  # only set if parse failed
    }


def save_batch(records, output_dir):
    """
    Saves all generated questions to a timestamped JSON file in ./questions/
    Returns the file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = os.path.join(output_dir, f"batch_{timestamp}.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    return filepath


def print_question(record, index):
    """Pretty-prints a question to the terminal so you can eyeball it."""
    print(f"\n{'─'*60}")
    print(f"  Q{index+1}. [{record['source']['filename']}  p.{record['source']['page']}]")
    print(f"  {record['question']}")
    if record.get("options"):
        for key, val in record["options"].items():
            marker = "✓" if key == record.get("correct_answer") else " "
            print(f"    {marker} {key}) {val}")
    print(f"\n  Explanation: {record.get('explanation', '—')}")
    print(f"  Cited: \"{record.get('cited_extract', '—')}\"")
    if record.get("parse_error"):
        print(f"  ⚠ PARSE ERROR — raw response saved to file for inspection.")


def main():
    # ── 1. Determine topics to generate questions for ─────────────────────────
    # If topics were passed as command-line args, use those.
    # Otherwise fall back to DEFAULT_TOPICS.
    if len(sys.argv) > 1:
        topics = sys.argv[1:]
        print(f"\nUsing {len(topics)} topic(s) from command line.")
    else:
        topics = DEFAULT_TOPICS
        print(f"\nNo topics specified — using {len(topics)} default topics.")

    print(f"Topics: {topics}")

    # ── 2. Check API key is set ───────────────────────────────────────────────
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Run:  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    # ── 3. Load ChromaDB collection ───────────────────────────────────────────
    print()
    collection = load_collection()

    # ── 4. Create Anthropic client ────────────────────────────────────────────
    anthropic_client = anthropic.Anthropic()

    # ── 5. Generate one question per topic ────────────────────────────────────
    records = []

    for i, topic in enumerate(topics):
        print(f"\n[{i+1}/{len(topics)}] Topic: '{topic}'")

        # Retrieve the best matching chunk from ChromaDB
        chunk = retrieve_best_chunk(collection, topic)
        print(f"  Best chunk: {chunk['source']} p.{chunk['page']} (distance={chunk['distance']})")

        # Call Claude API — generate question from chunk only
        print(f"  Calling Claude ({CLAUDE_MODEL}) ...")
        question_data = generate_question(anthropic_client, chunk)

        # Combine everything into one record
        record = build_output_record(topic, chunk, question_data)
        records.append(record)

        # Show a preview in the terminal
        print_question(record, i)

    # ── 6. Save all questions to file ─────────────────────────────────────────
    output_file = save_batch(records, OUTPUT_DIR)

    # ── 7. Summary ────────────────────────────────────────────────────────────
    ok      = sum(1 for r in records if not r.get("parse_error"))
    errored = len(records) - ok

    print(f"\n{'═'*52}")
    print(f"  GENERATION COMPLETE")
    print(f"  Questions generated  : {ok} / {len(records)}")
    print(f"  Parse errors         : {errored}")
    print(f"  Saved to             : {output_file}")
    print(f"{'═'*52}")
    print("\nNext step → run check.py to adversarially verify each question.")


if __name__ == "__main__":
    main()
