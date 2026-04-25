# fetch_current_affairs.py
# CSE-GO Pipeline — Current Affairs Ingestion
#
# Two modes:
#
#   A. PIB SCRAPER — pulls recent Press Information Bureau releases,
#      saves them to ./source-docs/ca/ as text files, then ingests
#      into a ChromaDB collection "current_affairs".
#
#   B. PDF DROP — point at any current affairs PDF (Vision IAS, Insights,
#      ForumIAS etc.) and it gets chunked + ingested the same way.
#
#   C. PASTE MODE — paste a news article directly; Claude generates MCQs
#      from it without touching ChromaDB at all.
#
# ── INSTALL ───────────────────────────────────────────────────────────────────
#   pip install requests feedparser chromadb sentence-transformers
#
# ── HOW TO RUN ────────────────────────────────────────────────────────────────
#
#   # Pull last 7 days of PIB releases and ingest:
#   python fetch_current_affairs.py --pib --days 7
#
#   # Ingest a CA PDF you've dropped in source-docs/ca/:
#   python fetch_current_affairs.py --pdf source-docs/ca/vision_ias_april_2026.pdf
#
#   # Generate MCQs from a pasted article (no ChromaDB needed):
#   python fetch_current_affairs.py --paste
#
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import time
import datetime
import argparse
import re

import requests
import feedparser
import fitz
import chromadb
import anthropic
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# ── CONFIG ────────────────────────────────────────────────────────────────────

CA_DIR       = "./source-docs/ca"
CHROMA_DIR   = "./chroma-db"
COLLECTION   = "current_affairs"
EMBED_MODEL  = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-sonnet-4-6"
OUTPUT_DIR   = "./questions"

CHUNK_WORDS  = 400
OVERLAP_WORDS = 40

# PIB RSS feeds — multiple ministries for broad UPSC coverage
PIB_FEEDS = [
    ("https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",  "PMO"),
    ("https://pib.gov.in/RssMain.aspx?ModId=2&Lang=1&Regid=3",  "Finance"),
    ("https://pib.gov.in/RssMain.aspx?ModId=21&Lang=1&Regid=3", "Environment"),
    ("https://pib.gov.in/RssMain.aspx?ModId=23&Lang=1&Regid=3", "Science_Technology"),
    ("https://pib.gov.in/RssMain.aspx?ModId=14&Lang=1&Regid=3", "External_Affairs"),
    ("https://pib.gov.in/RssMain.aspx?ModId=10&Lang=1&Regid=3", "Home"),
    ("https://pib.gov.in/RssMain.aspx?ModId=16&Lang=1&Regid=3", "Agriculture"),
    ("https://pib.gov.in/RssMain.aspx?ModId=20&Lang=1&Regid=3", "Health"),
]


# ── PIB SCRAPER ───────────────────────────────────────────────────────────────

def fetch_pib_article(url):
    """Fetch full text of a PIB article page."""
    try:
        r = requests.get(url, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"}, verify=False)
        r.raise_for_status()
        # Extract text from <div class="innner-page-main-about-us-content-right-part">
        # PIB pages have the article text in a specific div
        text = r.text
        # Simple extraction: find content between common markers
        patterns = [
            r'<div[^>]*class="[^"]*innner-page[^"]*"[^>]*>(.*?)</div>',
            r'<div[^>]*id="PCM"[^>]*>(.*?)</div>',
        ]
        for pat in patterns:
            m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
            if m:
                # Strip HTML tags
                content = re.sub(r'<[^>]+>', ' ', m.group(1))
                content = re.sub(r'&nbsp;', ' ', content)
                content = re.sub(r'\s+', ' ', content).strip()
                if len(content) > 100:
                    return content

        # Fallback: extract all paragraph text
        paras = re.findall(r'<p[^>]*>(.*?)</p>', text, re.DOTALL | re.IGNORECASE)
        clean = []
        for p in paras:
            t = re.sub(r'<[^>]+>', '', p).strip()
            if len(t) > 50:
                clean.append(t)
        return '\n\n'.join(clean) if clean else None

    except Exception as e:
        return None


def fetch_pib(days=7):
    """Pull recent PIB releases from RSS feeds. Returns list of {title, text, date, ministry, url}."""
    os.makedirs(CA_DIR, exist_ok=True)
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
    articles = []
    seen_urls = set()

    print(f"\nFetching PIB releases from last {days} days ...")

    for feed_url, ministry in PIB_FEEDS:
        print(f"  {ministry} ... ", end="", flush=True)
        try:
            import urllib3
            urllib3.disable_warnings()
            feed = feedparser.parse(feed_url)
            count = 0
            for entry in feed.entries:
                pub = entry.get("published_parsed")
                if pub:
                    pub_dt = datetime.datetime(*pub[:6])
                    if pub_dt < cutoff:
                        continue
                url = entry.get("link", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                title   = entry.get("title", "").strip()
                summary = re.sub(r'<[^>]+>', '', entry.get("summary", "")).strip()

                # Try fetching full text; fall back to summary
                full_text = fetch_pib_article(url) if url else None
                text = full_text if full_text and len(full_text) > len(summary) else summary

                if text and len(text.split()) > 30:
                    articles.append({
                        "title":    title,
                        "text":     text,
                        "date":     str(pub_dt.date()) if pub else "unknown",
                        "ministry": ministry,
                        "url":      url,
                        "source":   "PIB"
                    })
                    count += 1
                time.sleep(0.2)
            print(f"{count} articles")
        except Exception as e:
            print(f"error ({e})")

    print(f"  Total articles fetched: {len(articles)}")
    return articles


# ── PDF INGESTION ─────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path):
    """Extract text from a CA PDF, page by page."""
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text().strip()
        if text and len(text.split()) > 20:
            pages.append((i + 1, text))
    doc.close()
    return pages


# ── CHUNKER ───────────────────────────────────────────────────────────────────

def chunk_text(text, source_label, chunk_size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Split text into overlapping word chunks."""
    words = text.split()
    step  = chunk_size - overlap
    chunks = []
    for i in range(0, len(words), step):
        window = words[i:i + chunk_size]
        chunks.append({
            "text":   " ".join(window),
            "source": source_label,
        })
    return chunks


# ── CHROMA INGEST ─────────────────────────────────────────────────────────────

def ingest_to_chroma(chunks, collection):
    """Upsert chunks into the current_affairs ChromaDB collection."""
    docs  = [c["text"] for c in chunks]
    metas = [{"source": c["source"], "date": c.get("date",""), "ministry": c.get("ministry","")} for c in chunks]
    ids   = [f"ca_{i}_{abs(hash(c['text']))}" for i, c in enumerate(chunks)]

    batch = 100
    for start in range(0, len(docs), batch):
        collection.upsert(
            documents=docs[start:start+batch],
            metadatas=metas[start:start+batch],
            ids=ids[start:start+batch]
        )


# ── PASTE-AND-GENERATE ────────────────────────────────────────────────────────

PASTE_SYSTEM = """\
You are an expert UPSC Civil Services Examination question setter.

Given a news article or text, generate 2-3 high-quality UPSC Prelims MCQ questions.

Rules:
- Each question must be directly answerable from the article text.
- Do not use outside knowledge — only what's in the article.
- Match UPSC style: statement-based, assertion-reason, or single-fact formats.
- Return ONLY a valid JSON array of question objects, no markdown fences.

Each question object:
{
  "question": "...",
  "options": {"A":"...","B":"...","C":"...","D":"..."},
  "correct_answer": "A",
  "explanation": "...",
  "cited_extract": "verbatim sentence from article"
}\
"""

def paste_and_generate(client):
    """Interactive: user pastes article, Claude generates MCQs."""
    print("\nPaste your article/news text below.")
    print("When done, type END on a new line and press Enter:\n")

    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        except EOFError:
            break

    article = "\n".join(lines).strip()
    if not article:
        print("No text entered.")
        return

    print(f"\nGenerating MCQs from {len(article.split())} words ...")

    response = client.messages.create(
        model      = CLAUDE_MODEL,
        max_tokens = 2048,
        system     = PASTE_SYSTEM,
        messages   = [{"role": "user", "content": f"ARTICLE:\n\n{article}\n\nGenerate 2-3 UPSC MCQ questions from this article."}]
    )

    raw = response.content[0].text.strip()
    try:
        questions = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r'^```json|^```|```$', '', raw, flags=re.MULTILINE).strip()
        try:
            questions = json.loads(cleaned)
        except:
            print("\nCould not parse response. Raw output:")
            print(raw)
            return

    # Display
    for i, q in enumerate(questions):
        print(f"\n{'─'*60}")
        print(f"  Q{i+1}. {q.get('question','')}")
        for k in ["A","B","C","D"]:
            marker = "✓" if k == q.get("correct_answer") else " "
            print(f"    {marker} {k}) {q['options'].get(k,'')}")
        print(f"\n  Explanation: {q.get('explanation','')}")
        print(f"  Extract: \"{q.get('cited_extract','')[:100]}\"")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"ca_paste_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to: {path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Current affairs ingestion for CSE-GO")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pib",   action="store_true", help="Fetch PIB releases")
    group.add_argument("--pdf",   help="Path to a CA PDF to ingest")
    group.add_argument("--paste", action="store_true", help="Paste article → generate MCQs (no ChromaDB)")
    parser.add_argument("--days", type=int, default=7, help="Days back to fetch PIB (default 7)")
    args = parser.parse_args()

    # Paste mode needs API key, not ChromaDB
    if args.paste:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: ANTHROPIC_API_KEY not set.")
            sys.exit(1)
        paste_and_generate(anthropic.Anthropic())
        return

    # PIB and PDF modes need ChromaDB
    import urllib3
    urllib3.disable_warnings()

    client_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn      = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection    = client_chroma.get_or_create_collection(COLLECTION, embedding_function=embed_fn)
    existing      = collection.count()
    if existing:
        print(f"Collection '{COLLECTION}' already has {existing} chunks.")

    all_chunks = []

    if args.pib:
        articles = fetch_pib(days=args.days)
        for art in articles:
            label = f"PIB_{art['ministry']}_{art['date']}: {art['title'][:60]}"
            chunks = chunk_text(art["text"], label)
            for c in chunks:
                c["date"]     = art["date"]
                c["ministry"] = art["ministry"]
            all_chunks.extend(chunks)

    elif args.pdf:
        if not os.path.exists(args.pdf):
            print(f"ERROR: File not found: {args.pdf}")
            sys.exit(1)
        print(f"\nExtracting text from: {args.pdf}")
        pages = extract_pdf_text(args.pdf)
        print(f"  Pages with text: {len(pages)}")
        full_text = " ".join(text for _, text in pages)
        label = os.path.basename(args.pdf)
        all_chunks = chunk_text(full_text, label)

    if not all_chunks:
        print("No content to ingest.")
        return

    print(f"\nIngesting {len(all_chunks)} chunks into '{COLLECTION}' ...")
    ingest_to_chroma(all_chunks, collection)
    print(f"Done. Collection now has {collection.count()} chunks.")
    print(f"\nNext: run agent_generate.py — it will search current_affairs alongside NCERT.")


if __name__ == "__main__":
    main()
