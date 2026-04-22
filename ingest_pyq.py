# ingest_pyq.py
# CSE-GO Pipeline — Load parsed PYQ questions into ChromaDB
#
# Reads pyq_questions.json (output of parse_pyq.py) and stores each question
# in a separate ChromaDB collection "pyq_questions" so generate.py and future
# quiz scripts can retrieve real UPSC questions by topic similarity.
#
# Each stored document = question text + options (concatenated for embedding).
# Metadata stores year, subject, answer, explanation for retrieval.
#
# ── HOW TO RUN ────────────────────────────────────────────────────────────────
#   python ingest_pyq.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

JSON_PATH   = "./pyq_questions.json"
CHROMA_DIR  = "./chroma-db"
COLLECTION  = "pyq_questions"
EMBED_MODEL = "all-MiniLM-L6-v2"

BATCH_SIZE  = 100   # ChromaDB upsert in batches to avoid memory issues


def build_embed_text(record):
    """
    Combines question + options into one string for embedding.
    This lets semantic search find questions by topic even if the query
    doesn't match the exact wording.
    """
    parts = [record["question"]]
    for letter in ["a", "b", "c", "d"]:
        val = (record.get("options") or {}).get(letter, "")
        if val:
            parts.append(f"({letter}) {val}")
    return " ".join(parts)


def main():
    if not os.path.exists(JSON_PATH):
        print(f"\nERROR: {JSON_PATH} not found.")
        print("Run parse_pyq.py first.")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Filter out records without a question or answer (parser artifacts)
    records = [r for r in records if r.get("question") and r.get("answer")]

    print(f"\nLoaded {len(records)} PYQ records from {JSON_PATH}")

    print(f"Connecting to ChromaDB at '{CHROMA_DIR}' ...")
    client   = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embed_fn
    )

    existing = collection.count()
    if existing > 0:
        print(f"  Collection already has {existing} records — will upsert (add/update).")

    documents = []
    metadatas = []
    ids       = []

    for i, r in enumerate(records):
        # Unique ID: year_subject_qnum (or index for unknown-year questions)
        year_str = str(r["year"]) if r["year"] else "unk"
        subj_str = (r["subject"] or "general").replace(" ", "_")[:20]
        uid = f"pyq_{year_str}_{subj_str}_q{r['q_num']}_{i}"

        embed_text = build_embed_text(r)

        meta = {
            "year":        r["year"] or 0,
            "subject":     r["subject"] or "",
            "q_num":       r["q_num"],
            "answer":      r["answer"] or "",
            "source_page": r["source_page"] or 0,
            # Store explanation truncated — ChromaDB metadata has size limits
            "explanation": (r["explanation"] or "")[:800],
            # Store full options as a JSON string
            "options_json": json.dumps(r.get("options") or {}),
        }

        documents.append(embed_text)
        metadatas.append(meta)
        ids.append(uid)

    # Upsert in batches
    total = len(documents)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        collection.upsert(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end]
        )
        print(f"  Upserted {end}/{total} ...")

    final_count = collection.count()

    print(f"\n{'═'*52}")
    print(f"  INGEST COMPLETE")
    print(f"  Records stored       : {total}")
    print(f"  Total in collection  : {final_count}")
    print(f"  Collection name      : {COLLECTION}")
    print(f"  ChromaDB location    : {CHROMA_DIR}/")
    print(f"{'═'*52}")
    print("\nPYQ collection is ready. generate.py can now retrieve real UPSC questions.")


if __name__ == "__main__":
    main()
