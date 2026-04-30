"""
add_source.py — Add a PDF or URL to the CSE-GO knowledge pool.

Usage:
    python3 add_source.py --pdf path/to/file.pdf
    python3 add_source.py --pdf path/to/file.pdf --subject "Economy"
    python3 add_source.py --pdf path/to/file.pdf --start-page 5 --end-page 120
    python3 add_source.py --url https://example.com/article
    python3 add_source.py --url https://example.com/article --subject "Current Affairs"

Valid subjects: History, Geography, Polity, Economics, Science & Technology,
                Current Affairs, Art & Culture, Environment, General
"""

import argparse
import os
import re
import sys
import hashlib
import urllib.request
import html
from html.parser import HTMLParser

import fitz
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_DIR  = "./chroma-db"
COLLECTION  = "cse_knowledge_base"
CHUNK_WORDS = 800
OVERLAP     = 150
EMBED_MODEL = "all-MiniLM-L6-v2"

VALID_SUBJECTS = {
    "history", "geography", "polity", "economics", "science & technology",
    "current affairs", "art & culture", "environment", "general",
}


# ── TEXT EXTRACTION ───────────────────────────────────────────────────────────

def extract_pdf(path):
    doc = fitz.open(path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text().strip()
        if text:
            pages.append((i + 1, text))
    doc.close()
    return pages


class _TextExtractor(HTMLParser):
    """Minimal HTML → plain text stripper (no extra deps)."""
    SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside",
                 "noscript", "form", "button", "svg", "figure"}

    def __init__(self):
        super().__init__()
        self.chunks = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.SKIP_TAGS:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag.lower() in self.SKIP_TAGS:
            self._skip = max(0, self._skip - 1)

    def handle_data(self, data):
        if self._skip == 0:
            text = data.strip()
            if text:
                self.chunks.append(text)

    def get_text(self):
        return " ".join(self.chunks)


def extract_url(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    parser = _TextExtractor()
    parser.feed(html.unescape(raw))
    text = parser.get_text()
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text.split()) < 100:
        print("WARNING: Very little text extracted from URL. The page may be JS-rendered.")
    return [(1, text)]


# ── CHUNKING ──────────────────────────────────────────────────────────────────

def chunk_pages(pages):
    word_list = []
    for page_num, text in pages:
        for word in text.split():
            word_list.append((word, page_num))

    chunks = []
    step = CHUNK_WORDS - OVERLAP
    i = 0
    while i < len(word_list):
        window = word_list[i: i + CHUNK_WORDS]
        chunks.append({
            "text": " ".join(w for w, _ in window),
            "start_page": window[0][1],
            "chunk_index": len(chunks),
        })
        i += step
    return chunks


# ── SUBJECT INFERENCE ─────────────────────────────────────────────────────────

def infer_subject(name):
    n = name.lower()
    if any(x in n for x in ("history", "past", "heritage", "culture", "medieval", "ancient")):
        return "History"
    if any(x in n for x in ("geography", "climate", "river", "soil", "geograph")):
        return "Geography"
    if any(x in n for x in ("polity", "constitution", "governance", "civics", "political", "parliament")):
        return "Polity"
    if any(x in n for x in ("economy", "economics", "finance", "budget", "gdp", "trade")):
        return "Economics"
    if any(x in n for x in ("science", "technology", "biology", "chemistry", "physics", "space")):
        return "Science & Technology"
    if any(x in n for x in ("current", "news", "affairs", "monthly", "weekly")):
        return "Current Affairs"
    if any(x in n for x in ("environment", "ecology", "wildlife", "forest", "climate change")):
        return "Environment"
    if any(x in n for x in ("art", "culture", "dance", "music", "festival", "painting")):
        return "Art & Culture"
    return "General"


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Add a PDF or URL to the CSE-GO knowledge pool.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", metavar="PATH", help="Path to a PDF file")
    group.add_argument("--url", metavar="URL",  help="URL of a webpage to ingest")
    parser.add_argument("--subject", metavar="SUBJECT",
                        help="UPSC subject tag (e.g. 'Current Affairs', 'Economy')")
    parser.add_argument("--start-page", metavar="N", type=int, default=None,
                        help="First PDF page to ingest (1-indexed, default: 1)")
    parser.add_argument("--end-page", metavar="N", type=int, default=None,
                        help="Last PDF page to ingest (1-indexed, default: last page)")
    args = parser.parse_args()

    # ── Determine source name + extract pages ─────────────────────────────────
    if args.pdf:
        path = os.path.expanduser(args.pdf)
        if not os.path.isfile(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)
        source_name = os.path.basename(path)
        print(f"\nSource  : {source_name} (PDF)")

        # Peek at total pages before slicing
        _doc = fitz.open(path)
        total_pages = len(_doc)
        _doc.close()

        start = max(1, args.start_page or 1)
        end   = min(total_pages, args.end_page or total_pages)
        if start > end:
            print(f"ERROR: --start-page ({start}) is after --end-page ({end}).")
            sys.exit(1)

        page_range = f"pages {start}–{end}" if (start > 1 or end < total_pages) else "all pages"
        print(f"Pages   : {total_pages} total  →  ingesting {page_range}")
        print("Extracting text ...")
        pages = [(pn, t) for pn, t in extract_pdf(path) if start <= pn <= end]
    else:
        source_name = args.url
        print(f"\nSource  : {source_name} (URL)")
        print("Fetching and extracting text ...")
        try:
            pages = extract_url(args.url)
        except Exception as e:
            print(f"ERROR fetching URL: {e}")
            sys.exit(1)

    if not pages:
        print("ERROR: No text could be extracted. Exiting.")
        sys.exit(1)

    total_words = sum(len(p[1].split()) for p in pages)
    if not args.pdf:
        print(f"Pages   : {len(pages)}")
    print(f"Words   : {total_words:,}")

    # ── Subject ───────────────────────────────────────────────────────────────
    if args.subject:
        subject = args.subject.strip().title()
        if subject.lower() not in VALID_SUBJECTS:
            print(f"WARNING: '{subject}' is not a standard subject. Using as-is.")
    else:
        subject = infer_subject(source_name)
        print(f"Subject : {subject} (auto-detected; use --subject to override)")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    chunks = chunk_pages(pages)
    print(f"Chunks  : {len(chunks)}")

    # ── Connect to ChromaDB ───────────────────────────────────────────────────
    print(f"\nConnecting to ChromaDB at '{CHROMA_DIR}' ...")
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION, embedding_function=ef)

    before = collection.count()

    # ── Check for duplicates ──────────────────────────────────────────────────
    existing = collection.get(where={"source": {"$eq": source_name}}, limit=1)
    if existing and existing["ids"]:
        print(f"\nWARNING: '{source_name}' already exists in the collection.")
        answer = input("Re-ingest and overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    # ── Upsert ────────────────────────────────────────────────────────────────
    # Use a stable hash of source name so re-ingesting the same file overwrites cleanly
    source_hash = hashlib.md5(source_name.encode()).hexdigest()[:8]

    documents, metadatas, ids = [], [], []
    for c in chunks:
        chunk_id = f"{source_hash}_p{c['start_page']}_c{c['chunk_index']}"
        documents.append(c["text"])
        metadatas.append({
            "source":      source_name,
            "page":        c["start_page"],
            "chunk_index": c["chunk_index"],
            "subject":     subject,
        })
        ids.append(chunk_id)

    print("Embedding and storing chunks ...")
    # Upsert in batches of 100 to avoid memory spikes
    batch = 100
    for i in range(0, len(chunks), batch):
        collection.upsert(
            documents=documents[i:i+batch],
            metadatas=metadatas[i:i+batch],
            ids=ids[i:i+batch],
        )
        print(f"  {min(i+batch, len(chunks))}/{len(chunks)} chunks stored ...", end="\r")

    after    = collection.count()
    net      = after - before
    text_mb  = sum(len(d.encode()) for d in documents) / (1024 * 1024)

    page_info = f"{page_range}" if args.pdf else f"{len(pages)} page(s)"

    print(f"\n\n{'═'*52}")
    print(f"  ✓ INGEST COMPLETE")
    print(f"  Source          : {source_name}")
    print(f"  Pages ingested  : {page_info}")
    print(f"  Subject         : {subject}")
    print(f"  Words ingested  : {total_words:,}")
    print(f"  Text size       : {text_mb:.2f} MB")
    print(f"  Chunks stored   : {len(chunks)}  (net new in DB: {net:+d})")
    print(f"  Collection total: {after:,} chunks")
    print(f"{'═'*52}")
    print("\nNext pipeline run will retrieve from this source automatically.")


if __name__ == "__main__":
    main()
