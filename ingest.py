# ingest.py
# CSE-GO Pipeline — Step 1: Extract → Chunk → Embed → Store
#
# What this file does:
#   Reads every PDF in /source-docs, breaks the text into overlapping
#   chunks, converts each chunk to a vector embedding, and saves
#   everything into a local ChromaDB database on your machine.
#
# ── INSTALL (run once in your terminal before running this script) ───────────
#
#   pip install pymupdf chromadb sentence-transformers
#
#   pymupdf            → extracts text from PDF files (imported as 'fitz')
#   chromadb           → local vector database; saves to disk, zero cloud setup
#   sentence-transformers → converts text into embeddings (numbers that capture meaning)
#
# ── HOW TO RUN ───────────────────────────────────────────────────────────────
#
#   python ingest.py
#
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import fitz  # PyMuPDF is imported as 'fitz' — this is just its internal package name
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Change these if your folder names are different

SOURCE_DIR    = "./source-docs"        # put your PDFs here
CHROMA_DIR    = "./chroma-db"          # ChromaDB will create and manage this folder
COLLECTION    = "cse_knowledge_base"   # name of the vector database collection
CHUNK_WORDS   = 500                    # each chunk is ~500 words
OVERLAP_WORDS = 50                     # last 50 words of chunk N become first 50 of chunk N+1
                                       # overlap preserves context at chunk boundaries
EMBED_MODEL   = "all-MiniLM-L6-v2"    # free, fast, runs locally — no API key needed
# ─────────────────────────────────────────────────────────────────────────────


# ── CHAPTER FILTER ───────────────────────────────────────────────────────────
# Pages to skip — these patterns appear on non-chapter pages in NCERT PDFs
SKIP_PHRASES = [
    'table of contents', 'foreword', 'preface', 'acknowledgement',
    'first edition', 'isbn ', 'reprint 20', 'printed by', 'published at',
    'national council of educational research', 'textbook development',
    'not for sale', 'bibliography', 'suggested readings', 'further reading',
    'answer key', 'answers to activities', 'answers to in-text',
]

# A chapter heading looks like: "Chapter 1", "CHAPTER 2", "Unit 3", "UNIT I"
CHAPTER_RE = re.compile(
    r'(chapter|unit)\s*[\d]+|(chapter|unit)\s*[ivxlcdm]+\b',
    re.IGNORECASE
)

def is_skip_page(text):
    """
    Returns True if this page should be excluded.
    Catches: title pages, copyright, TOC, foreword, preface,
             index, bibliography, answer keys.
    """
    # Too few words — title page, blank page, image-only page
    if len(text.split()) < 50:
        return True

    # Check first 400 characters for metadata phrases
    text_start = text[:400].lower()
    if any(phrase in text_start for phrase in SKIP_PHRASES):
        return True

    # Detect TOC pages: >50% of non-empty lines end with a digit (page numbers)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) >= 6:
        ends_with_digit = sum(1 for l in lines if l[-1].isdigit())
        if ends_with_digit / len(lines) > 0.5:
            return True

    return False


def filter_to_chapter_pages(pages):
    """
    From the full list of (page_num, text) tuples, return only pages that
    contain actual chapter content.

    NCERT chapter pages do NOT say "Chapter 1" at the top — they start
    directly with the chapter title (e.g. "MATTER IN OUR SURROUNDINGS").
    The word "Chapter" only appears in the Table of Contents.

    So the strategy is simpler:
      1. Skip any page that matches is_skip_page() (TOC, copyright, foreword etc.)
      2. Stop if we hit end-matter (Index, Bibliography, Answers).
      3. Keep everything else — those are chapter content pages.
    """
    END_MATTER_RE = re.compile(
        r'^(index|bibliography|appendix|glossary|answers\s+to)',
        re.IGNORECASE
    )

    result = []

    for page_num, text in pages:
        first_line = text.strip().split('\n')[0].strip()

        # Stop when we hit end-matter (index, bibliography, answers)
        if END_MATTER_RE.match(first_line):
            break

        # Drop junk pages (metadata, copyright, TOC, blank-ish)
        if is_skip_page(text):
            continue

        # Everything else is chapter content
        result.append((page_num, text))

    return result


def extract_pages(pdf_path):
    """
    Opens a PDF and returns a list of (page_number, text) tuples.
    Page numbers start at 1 (like a real book, not 0).
    Blank pages are skipped automatically.
    """
    doc = fitz.open(pdf_path)
    pages = []

    for i in range(len(doc)):
        text = doc[i].get_text()   # extract all plain text from this page
        text = text.strip()
        if text:                   # only keep pages that have actual text
            pages.append((i + 1, text))

    doc.close()
    return pages


def build_word_list(pages):
    """
    Converts a list of (page_num, text) into a flat list of (word, page_num) tuples.

    Why? So when we slice a 500-word window, we can always look up
    which page the first word of that window came from.

    Example output:
        [("The", 1), ("Constitution", 1), ..., ("Parliament", 2), ...]
    """
    word_list = []
    for page_num, text in pages:
        words = text.split()              # split on whitespace
        for word in words:
            word_list.append((word, page_num))
    return word_list


def create_chunks(word_list, chunk_size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """
    Slides a window of `chunk_size` words across the word list.
    Each step advances by (chunk_size - overlap) words so the next chunk
    starts `overlap` words before the previous one ended.

    Returns a list of dicts:
        {
          "text":        "the full chunk as a string",
          "start_page":  page number where this chunk begins,
          "chunk_index": 0, 1, 2, ... (sequential within this document)
        }
    """
    chunks = []
    step = chunk_size - overlap    # how far to move the window forward each time
    i = 0
    chunk_index = 0

    while i < len(word_list):
        window = word_list[i : i + chunk_size]      # grab the next window of words
        text = " ".join(w for w, _ in window)       # join words back into readable text
        start_page = window[0][1]                   # page number of the first word

        chunks.append({
            "text": text,
            "start_page": start_page,
            "chunk_index": chunk_index
        })

        chunk_index += 1
        i += step                                   # move forward (not full chunk — overlap!)

    return chunks


def ingest_one_pdf(pdf_path, collection):
    """
    Runs the full extract → chunk → store pipeline for a single PDF.
    Returns the number of chunks stored.
    """
    filename = os.path.basename(pdf_path)
    print(f"\n  Processing: {filename}")

    # Extract text page by page
    all_pages = extract_pages(pdf_path)
    if not all_pages:
        print(f"  WARNING: No text found in {filename}. Is it a scanned image PDF?")
        print(f"           Skipping — only text-based PDFs work with this pipeline.")
        return 0

    # Keep only actual chapter content — drop TOC, copyright, foreword, index etc.
    pages = filter_to_chapter_pages(all_pages)
    if not pages:
        print(f"  WARNING: No chapter content detected in {filename} — skipping.")
        return 0

    # Flatten into (word, page_num) pairs
    word_list = build_word_list(pages)
    print(f"    Pages (total)        : {len(all_pages)}")
    print(f"    Pages (chapter only) : {len(pages)}  ({len(all_pages)-len(pages)} non-chapter pages dropped)")
    print(f"    Total words     : {len(word_list)}")

    # Slice into overlapping chunks
    chunks = create_chunks(word_list)
    print(f"    Chunks created  : {len(chunks)}")

    # Build the three lists ChromaDB needs:
    #   documents → the actual text of each chunk
    #   metadatas → a dict of extra info stored alongside each chunk
    #   ids       → a unique string ID for each chunk (required by ChromaDB)
    documents = []
    metadatas = []
    ids       = []

    for chunk in chunks:
        # Build a unique ID from filename + page + chunk number
        # e.g. "NCERT_Geography_Class10.pdf_p3_c7"
        chunk_id = f"{filename}_p{chunk['start_page']}_c{chunk['chunk_index']}"

        documents.append(chunk["text"])
        metadatas.append({
            "source":      filename,
            "page":        chunk["start_page"],
            "chunk_index": chunk["chunk_index"]
        })
        ids.append(chunk_id)

    # upsert = insert if new, update if already exists
    # This makes it safe to run ingest.py multiple times without duplicate errors
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    return len(chunks)


def main():
    # ── Check that source-docs folder exists ─────────────────────────────────
    if not os.path.exists(SOURCE_DIR):
        print(f"\nERROR: '{SOURCE_DIR}' folder not found.")
        print("Create it in the same folder as ingest.py and add your PDFs.")
        return

    # ── Collect all PDF paths ─────────────────────────────────────────────────
    pdf_files = sorted([
        os.path.join(SOURCE_DIR, f)
        for f in os.listdir(SOURCE_DIR)
        if f.lower().endswith(".pdf")
    ])

    if not pdf_files:
        print(f"\nNo PDF files found in '{SOURCE_DIR}'.")
        print("Add your NCERT PDFs and previous year papers, then run again.")
        return

    print(f"\nFound {len(pdf_files)} PDF(s) in {SOURCE_DIR}:")
    for p in pdf_files:
        print(f"  • {os.path.basename(p)}")

    # ── Connect to ChromaDB ───────────────────────────────────────────────────
    # PersistentClient saves the database to the CHROMA_DIR folder on disk.
    # This means your embeddings survive between runs — you don't re-embed every time.
    print(f"\nConnecting to ChromaDB at '{CHROMA_DIR}' ...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Tell ChromaDB which model to use for converting text → embeddings
    # It will download the model the first time (a few seconds), then cache it locally
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    # Get the collection if it already exists, or create it fresh
    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embed_fn
    )

    existing_count = collection.count()
    if existing_count > 0:
        print(f"  Collection already has {existing_count} chunks — new chunks will be added/updated.")

    # ── Ingest each PDF ───────────────────────────────────────────────────────
    total_chunks  = 0
    pdfs_ok       = 0

    for pdf_path in pdf_files:
        n = ingest_one_pdf(pdf_path, collection)
        total_chunks += n
        if n > 0:
            pdfs_ok += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 52)
    print("  INGEST COMPLETE")
    print(f"  PDFs processed       : {pdfs_ok} / {len(pdf_files)}")
    print(f"  Chunks stored        : {total_chunks}")
    print(f"  Total in collection  : {collection.count()}")
    print(f"  ChromaDB location    : {CHROMA_DIR}/")
    print(f"  Collection name      : {COLLECTION}")
    print("═" * 52)
    print("\nNext step → run generate.py to retrieve chunks and generate questions.")


if __name__ == "__main__":
    main()
