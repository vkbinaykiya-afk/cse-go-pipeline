# ingest_ca.py
# CSE-GO Pipeline — Current Affairs Ingestion with Reword Agent
#
# Processes CA sources through a reword agent before storage:
#   Raw text → Claude rewrites in own words → extracts UPSC-testable facts
#   → stores synthesized version in ChromaDB "current_affairs" collection
#
# Facts are preserved. Source phrasing is not. Stored text is Claude's
# synthesis, not the original author's expression.
# All reworded chunks are flagged "pending_verification" until reviewed.
#
# ── SOURCES SUPPORTED ─────────────────────────────────────────────────────────
#
#   PIB  — Press Information Bureau RSS (government schemes, policies)
#   PDF  — Any CA compilation PDF dropped in source-docs/ca/
#   Text — Raw text piped from stdin or a .txt file
#
# ── INSTALL ───────────────────────────────────────────────────────────────────
#   pip install requests feedparser pymupdf chromadb sentence-transformers anthropic
#
# ── HOW TO RUN ────────────────────────────────────────────────────────────────
#
#   # Pull PIB releases from last 14 days:
#   python ingest_ca.py --pib --days 14
#
#   # Ingest a monthly CA PDF:
#   python ingest_ca.py --pdf source-docs/ca/insights_april_2026.pdf
#
#   # Quick paste-and-generate (no ChromaDB, direct MCQ output):
#   python ingest_ca.py --paste
#
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import time
import datetime
import argparse
import re
import hashlib

import requests
import feedparser
import fitz
import chromadb
import anthropic
import urllib3
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

urllib3.disable_warnings()

# ── CONFIG ────────────────────────────────────────────────────────────────────

CA_SOURCE_DIR = "./source-docs/ca"
CHROMA_DIR    = "./chroma-db"
CA_COLLECTION = "current_affairs"
EMBED_MODEL   = "all-MiniLM-L6-v2"

REWORD_MODEL  = "claude-haiku-4-5-20251001"   # cheaper/faster for rewriting
GENERATE_MODEL= "claude-sonnet-4-6"            # kept for MCQ generation

CHUNK_WORDS   = 450
OVERLAP_WORDS = 45
BATCH_SIZE    = 50

PIB_FEEDS = [
    ("https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",  "PMO"),
    ("https://pib.gov.in/RssMain.aspx?ModId=2&Lang=1&Regid=3",  "Finance"),
    ("https://pib.gov.in/RssMain.aspx?ModId=21&Lang=1&Regid=3", "Environment"),
    ("https://pib.gov.in/RssMain.aspx?ModId=23&Lang=1&Regid=3", "Science_Technology"),
    ("https://pib.gov.in/RssMain.aspx?ModId=14&Lang=1&Regid=3", "External_Affairs"),
    ("https://pib.gov.in/RssMain.aspx?ModId=16&Lang=1&Regid=3", "Agriculture"),
    ("https://pib.gov.in/RssMain.aspx?ModId=20&Lang=1&Regid=3", "Health"),
    ("https://pib.gov.in/RssMain.aspx?ModId=10&Lang=1&Regid=3", "Home"),
]


# ── REWORD AGENT ──────────────────────────────────────────────────────────────

REWORD_SYSTEM = """\
You are a UPSC Civil Services Examination expert. Your job is to synthesize raw \
current affairs text into a study-ready form that a question-setter can directly use \
to write Prelims MCQs.

UPSC does NOT test breaking news or raw facts in isolation. It tests whether a student \
can CONNECT a current event to an underlying concept, constitutional provision, \
institutional framework, or historical pattern. Every fact you extract must carry that \
connection angle.

── STEP 0 — SUBSTANCE CHECK ─────────────────────────────────────────────────────
First, decide if the chunk is substantive prose. Skip it if it is:
  • A table of contents, index, or list of topic headings
  • A page of bullet labels without explanatory sentences (e.g. "X vs Y", "A: ..., B: ...")
  • A glossary, bibliography, or question-answer list with no context
  • Fewer than 4 complete declarative sentences

If the chunk fails the substance check, return ONLY:
  {"skip": true, "reason": "one-line explanation"}

── STEP 1 — REWRITE ─────────────────────────────────────────────────────────────
Rewrite the text ENTIRELY in your own words. Change sentence structure, vocabulary, \
and phrasing. Do not copy any phrase of 5+ consecutive words from the source. \
Preserve every factual claim exactly (names, numbers, dates, provisions).

OUTPUT STYLE RULES for reworded_text — violations will cause the chunk to be discarded:
  • Every sentence must be a complete declarative statement (subject + verb + object).
  • NO headings, NO bullet labels, NO "X vs Y" constructs, NO numbered lists.
  • NO sentences that are just topic names or category labels.
  • If you cannot write 2+ full paragraphs of declarative prose from this chunk, \
return {"skip": true, "reason": "insufficient substantive content"}.

── STEP 2 — EXTRACT UPSC-TESTABLE FACTS ─────────────────────────────────────────
Extract 3-6 facts. For each fact, identify:
  • The raw fact (what happened / what exists)
  • The UPSC concept it connects to (the "why it's testable")

Categories that UPSC most commonly tests from current affairs:
  SCHEMES     — name, nodal ministry, target beneficiary, funding pattern, year launched
  ORGS        — full name, mandate, HQ, India's role/membership, year founded
  TREATIES    — parties, key obligations, year, India's status (signed/ratified/not party)
  SPECIES     — scientific name if notable, IUCN status, schedule under Wildlife Act,
                  habitat/range, endemic or not
  GEOGRAPHY   — river basin, mountain range, national park, biosphere reserve; key facts
  POLITY      — constitutional article, schedule, amendment; which list (Union/State/Concurrent)
  ECONOMY     — index name + publisher, India's rank/score, what it measures
  ENVIRONMENT — convention, protocol, body; India's commitments; key targets/timelines
  SCIENCE     — mission/programme name, agency, objective, India-specific angle
  HISTORY     — event → constitutional or institutional outcome still relevant today

── STEP 3 — CONCEPT LINKS ────────────────────────────────────────────────────────
For each extracted fact, add a short "concept_link" note: the textbook / static \
concept a student must know to answer a question based on this fact. E.g.:
  Fact: "India ratified the Minamata Convention in 2018."
  concept_link: "Conventions under UNEP; mercury pollution; Basel/Rotterdam/Stockholm \
family of conventions"

── OUTPUT FORMAT ─────────────────────────────────────────────────────────────────
Return ONLY valid JSON — no markdown fences, no explanation outside the JSON:
{
  "reworded_text": "Full synthesis here. 2-4 paragraphs. Reads like a concise \
briefing note, not a news article.",
  "topic": "Short label — scheme/org/event name, e.g. 'PM-KUSUM Scheme' or \
'Ramsar Sites India 2026'",
  "category": "one of: SCHEMES | ORGS | TREATIES | SPECIES | GEOGRAPHY | POLITY \
| ECONOMY | ENVIRONMENT | SCIENCE | HISTORY | OTHER",
  "upsc_facts": [
    {
      "fact": "Complete sentence stating the testable fact.",
      "concept_link": "The underlying static concept this connects to."
    }
  ],
  "verification_status": "pending"
}\
"""

def _is_substantive(text):
    """
    Local pre-filter before hitting the API.
    Returns False if the chunk looks like an index, TOC, or heading list.
    """
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 20]
    if len(sentences) < 3:
        return False

    words = text.split()
    # High ratio of short lines = heading/bullet list
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        short_lines = sum(1 for l in lines if len(l.split()) < 6)
        if short_lines / len(lines) > 0.6:
            return False

    # Mostly dots/ellipses = TOC page
    if text.count('...') + text.count('….') > 5:
        return False

    return True


def reword_chunk(client, raw_text, source_label):
    """
    Send a raw text chunk through the reword agent.
    Returns a dict with reworded_text, topic, upsc_facts, verification_status,
    or None if the chunk should be skipped (non-substantive).
    Falls back to storing raw text if API fails.
    """
    # Local pre-filter — skip obvious TOC/index pages without an API call
    if not _is_substantive(raw_text):
        return None

    try:
        response = client.messages.create(
            model      = REWORD_MODEL,
            max_tokens = 1024,
            system     = [{"type": "text", "text": REWORD_SYSTEM,
                           "cache_control": {"type": "ephemeral"}}],
            messages   = [{"role": "user", "content":
                           f"SOURCE: {source_label}\n\nTEXT:\n{raw_text}"}]
        )
        raw = response.content[0].text.strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            cleaned = re.sub(r'^```json|^```|```$', '', raw, flags=re.MULTILINE).strip()
            result  = json.loads(cleaned)

        # Honour model's own skip signal
        if result.get("skip"):
            return None

        # Reject if reworded_text looks like headings (no real sentences)
        reworded = result.get("reworded_text", "")
        if not _is_substantive(reworded):
            return None

        return result

    except Exception as e:
        # Fallback: store raw text, flag as reword_failed
        return {
            "reworded_text":       raw_text,
            "topic":               source_label[:80],
            "category":            "OTHER",
            "upsc_facts":          [],
            "verification_status": "reword_failed"
        }


# ── TEXT EXTRACTION ───────────────────────────────────────────────────────────

def chunk_words(text, chunk_size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Split text into overlapping word-count chunks."""
    words = text.split()
    step  = chunk_size - overlap
    chunks = []
    for i in range(0, len(words), step):
        window = words[i:i + chunk_size]
        if len(window) >= 40:   # skip tiny trailing chunks
            chunks.append(" ".join(window))
    return chunks


def extract_pdf_chunks(pdf_path):
    """Extract text from PDF and return raw chunks with page info."""
    doc    = fitz.open(pdf_path)
    pages  = []
    for i in range(len(doc)):
        text = doc[i].get_text().strip()
        if text and len(text.split()) > 30:
            pages.append((i + 1, text))
    doc.close()

    full_text = " ".join(t for _, t in pages)
    return chunk_words(full_text), len(pages)


def fetch_pib_articles(days=7):
    """Pull recent PIB releases. Returns list of {title, text, date, ministry}."""
    cutoff   = datetime.datetime.now() - datetime.timedelta(days=days)
    articles = []
    seen     = set()

    print(f"\nFetching PIB — last {days} days:")
    for feed_url, ministry in PIB_FEEDS:
        print(f"  {ministry:<20}", end="", flush=True)
        try:
            feed  = feedparser.parse(feed_url)
            count = 0
            for entry in feed.entries:
                pub = entry.get("published_parsed")
                if pub and datetime.datetime(*pub[:6]) < cutoff:
                    continue
                url = entry.get("link","")
                if url in seen:
                    continue
                seen.add(url)

                title   = entry.get("title","").strip()
                summary = re.sub(r'<[^>]+>', '', entry.get("summary","")).strip()
                date    = str(datetime.datetime(*pub[:6]).date()) if pub else "unknown"

                if summary and len(summary.split()) > 30:
                    articles.append({
                        "title":    title,
                        "text":     f"{title}. {summary}",
                        "date":     date,
                        "ministry": ministry,
                    })
                    count += 1
                time.sleep(0.1)
            print(f"{count} articles")
        except Exception as e:
            print(f"error — {e}")

    print(f"  Total: {len(articles)} articles")
    return articles


# ── CHROMA INGEST ─────────────────────────────────────────────────────────────

def ingest_reworded(reworded_chunks, collection):
    """Upsert reworded chunks into ChromaDB."""
    documents = []
    metadatas = []
    ids       = []

    for item in reworded_chunks:
        text = item["reworded_text"]
        uid  = f"ca_{hashlib.md5(text.encode()).hexdigest()[:16]}"

        documents.append(text)
        metadatas.append({
            "topic":               item.get("topic","")[:200],
            "category":            item.get("category","OTHER"),
            "upsc_facts":          json.dumps(item.get("upsc_facts",[])),
            "verification_status": item.get("verification_status","pending"),
            "source":              item.get("source","")[:200],
            "date":                item.get("date",""),
        })
        ids.append(uid)

    for start in range(0, len(documents), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(documents))
        collection.upsert(
            documents = documents[start:end],
            metadatas = metadatas[start:end],
            ids       = ids[start:end]
        )

    return len(documents)


# ── PASTE-AND-GENERATE ────────────────────────────────────────────────────────

PASTE_SYSTEM = """\
You are an expert UPSC Prelims question setter. Given a news article,
generate 2-3 UPSC-style MCQ questions. The answer must be directly
supportable from the article text. Use UPSC question formats:
statement-based ("how many are correct"), assertion-reason, or single-fact.

Return ONLY a valid JSON array — no markdown, no explanation:
[
  {
    "question": "...",
    "options": {"A":"...","B":"...","C":"...","D":"..."},
    "correct_answer": "A",
    "explanation": "...",
    "cited_extract": "verbatim sentence from article"
  }
]\
"""

def paste_and_generate(client):
    print("\nPaste article text. Type END on a new line when done:\n")
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

    print(f"\nGenerating MCQs from {len(article.split())} words ...\n")

    r = client.messages.create(
        model    = GENERATE_MODEL,
        max_tokens = 2048,
        system   = PASTE_SYSTEM,
        messages = [{"role":"user","content": f"ARTICLE:\n\n{article}\n\nGenerate 2-3 UPSC MCQ questions."}]
    )

    raw = r.content[0].text.strip()
    try:
        questions = json.loads(raw)
    except:
        cleaned = re.sub(r'^```json|^```|```$', '', raw, flags=re.MULTILINE).strip()
        try:
            questions = json.loads(cleaned)
        except:
            print("Could not parse response:\n", raw)
            return

    for i, q in enumerate(questions):
        print(f"{'─'*60}")
        print(f"Q{i+1}. {q.get('question','')}")
        for k in ["A","B","C","D"]:
            marker = "✓" if k == q.get("correct_answer") else " "
            print(f"  {marker} {k}) {q['options'].get(k,'')}")
        print(f"\n  Explanation: {q.get('explanation','')}")
        print(f"  Source: \"{q.get('cited_extract','')[:100]}\"\n")

    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("./questions", f"ca_paste_{ts}.json")
    os.makedirs("./questions", exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CA ingestion with reword agent")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pib",   action="store_true", help="Fetch PIB releases")
    group.add_argument("--pdf",   help="Path to a CA PDF")
    group.add_argument("--paste", action="store_true", help="Paste article → MCQs directly")
    parser.add_argument("--days", type=int, default=7, help="Days back for PIB (default 7)")
    parser.add_argument("--no-reword", action="store_true",
                        help="Skip reword agent — store raw text (faster, no API cost)")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    ai_client = anthropic.Anthropic()

    # ── Paste mode — no ChromaDB needed ──────────────────────────────────────
    if args.paste:
        paste_and_generate(ai_client)
        return

    # ── ChromaDB setup ────────────────────────────────────────────────────────
    chroma    = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn  = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = chroma.get_or_create_collection(CA_COLLECTION, embedding_function=embed_fn)
    print(f"\nChromaDB '{CA_COLLECTION}': {collection.count()} existing chunks")

    # ── Collect raw chunks ────────────────────────────────────────────────────
    raw_items = []   # list of {text, source, date}

    if args.pib:
        articles = fetch_pib_articles(days=args.days)
        for art in articles:
            label = f"PIB/{art['ministry']} {art['date']} — {art['title'][:50]}"
            for chunk_text in chunk_words(art["text"]):
                raw_items.append({
                    "text":   chunk_text,
                    "source": label,
                    "date":   art["date"],
                })

    elif args.pdf:
        if not os.path.exists(args.pdf):
            print(f"ERROR: {args.pdf} not found.")
            sys.exit(1)
        print(f"\nExtracting: {args.pdf}")
        raw_chunks, n_pages = extract_pdf_chunks(args.pdf)
        label = os.path.basename(args.pdf)
        print(f"  {n_pages} pages → {len(raw_chunks)} chunks")
        for chunk_text in raw_chunks:
            raw_items.append({
                "text":   chunk_text,
                "source": label,
                "date":   datetime.date.today().isoformat(),
            })

    if not raw_items:
        print("Nothing to ingest.")
        return

    print(f"\n{len(raw_items)} raw chunks ready.")

    # ── Reword agent ──────────────────────────────────────────────────────────
    reworded = []

    if args.no_reword:
        print("Skipping reword — storing raw text.")
        for item in raw_items:
            reworded.append({
                "reworded_text":     item["text"],
                "topic":             item["source"][:80],
                "upsc_facts":        [],
                "verification_status": "not_reworded",
                "source":            item["source"],
                "date":              item["date"],
            })
    else:
        print(f"Running reword agent ({REWORD_MODEL}) on {len(raw_items)} chunks ...")
        print("(Each dot = 1 chunk processed)\n", end="", flush=True)

        skipped = 0
        for i, item in enumerate(raw_items):
            result = reword_chunk(ai_client, item["text"], item["source"])

            if result is None:
                skipped += 1
                print("s", end="", flush=True)  # 's' = skipped (non-substantive)
            else:
                result["source"] = item["source"]
                result["date"]   = item["date"]
                reworded.append(result)
                print(".", end="", flush=True)

            if (i + 1) % 50 == 0:
                print(f" {i+1}/{len(raw_items)}")

            time.sleep(0.1)   # rate limit courtesy pause

        print(f"\n\nReword complete.")
        if skipped:
            print(f"  {skipped} chunks skipped (non-substantive: TOC, index, headings).")
        failed = sum(1 for r in reworded if r.get("verification_status") == "reword_failed")
        if failed:
            print(f"  {failed} chunks fell back to raw text (reword_failed).")

    # ── Store in ChromaDB ─────────────────────────────────────────────────────
    print(f"\nIngesting {len(reworded)} chunks into '{CA_COLLECTION}' ...")
    stored = ingest_reworded(reworded, collection)
    print(f"Done. '{CA_COLLECTION}' now has {collection.count()} chunks.")

    # ── Summary of extracted facts ────────────────────────────────────────────
    all_facts = [f for r in reworded for f in r.get("upsc_facts", [])]
    pending   = sum(1 for r in reworded if r.get("verification_status") == "pending")

    print(f"\n{'═'*52}")
    print(f"  INGEST COMPLETE")
    print(f"  Chunks stored        : {stored}")
    print(f"  UPSC facts extracted : {len(all_facts)}")
    print(f"  Pending verification : {pending}")
    print(f"{'═'*52}")
    print("\nSample facts extracted:")
    for fact in all_facts[:8]:
        print(f"  • {fact}")

    print(f"\nNext: agent_generate.py will now search current_affairs alongside NCERT.")


if __name__ == "__main__":
    main()
