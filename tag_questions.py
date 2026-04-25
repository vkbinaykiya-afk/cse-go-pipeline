"""
tag_questions.py — Batch-tag all untagged questions in the DB using Claude Haiku.

Adds:
  upsc_subject     : Polity | History | Geography | Economy | Environment |
                     Science & Tech | Current Affairs | Art & Culture
  upsc_topic       : UPSC syllabus-level topic (e.g. "Fundamental Rights")
  broad_category   : Parent grouping (e.g. "Part III of Constitution")
  question_category: factual | conceptual | trend_based | in_news | map_based

Usage:
    python3 tag_questions.py            # tag all untagged
    python3 tag_questions.py --retag    # retag everything
"""

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

import anthropic

DB_PATH = Path(__file__).parent / "cse_go.db"
MODEL = "claude-haiku-4-5-20251001"
BATCH = 5   # questions per API call

STATIC_SUBJECTS = [
    "History", "Geography", "Polity", "Environment",
    "Science & Technology", "Economics", "Current Affairs"
]

# Category options per subject
SUBJECT_CATEGORIES = {
    "History":              ["Ancient", "Medieval", "Modern"],
    "Geography":            ["Physical", "Economic", "Social", "World Geography"],
    "Polity":               ["Constitution", "Governance", "Parliament", "Judiciary", "Local Government", "Constitutional Bodies"],
    "Environment":          ["Biodiversity", "Climate Change", "Pollution & Ecology", "Conservation", "Disaster Management"],
    "Science & Technology": ["Space & ISRO", "Defence Technology", "IT & Cyber", "Biotechnology", "Energy", "General Science"],
    "Economics":            ["Macroeconomics", "Agriculture", "Infrastructure", "Industry & Trade", "Social Sector", "Public Finance"],
    "Current Affairs":      ["International Relations", "Governance & Policy", "Economy", "Environment", "Science", "Security", "Society"],
}

TAGGER_SYSTEM = f"""\
You are a UPSC (Union Public Service Commission) syllabus expert.
Given a set of MCQ questions, classify each one with:

1. upsc_subject — MUST be exactly one of:
   History | Geography | Polity | Environment | Science & Technology | Economics | Current Affairs
   Map "Art & Culture" questions to History. Map "Economy" to Economics. Map "Science & Tech" to Science & Technology.

2. upsc_category — the sub-category within the subject. Use these options:
   History:              Ancient | Medieval | Modern
   Geography:            Physical | Economic | Social | World Geography
   Polity:               Constitution | Governance | Parliament | Judiciary | Local Government | Constitutional Bodies
   Environment:          Biodiversity | Climate Change | Pollution & Ecology | Conservation | Disaster Management
   Science & Technology: Space & ISRO | Defence Technology | IT & Cyber | Biotechnology | Energy | General Science
   Economics:            Macroeconomics | Agriculture | Infrastructure | Industry & Trade | Social Sector | Public Finance
   Current Affairs:      International Relations | Governance & Policy | Economy | Environment | Science | Security | Society

3. upsc_topic — the specific UPSC syllabus topic this question tests.
   Examples: "Fundamental Rights", "Ramsar Convention", "Revolt of 1857", "Monetary Policy"
   For Current Affairs questions: prefix with "Current Affairs — " e.g. "Current Affairs — India-Bangladesh Relations"

4. question_category — one of:
   factual | conceptual | trend_based | in_news | map_based

Return ONLY a JSON array, one object per question, in the same order received:
[
  {{
    "id": "<question id>",
    "upsc_subject": "...",
    "upsc_category": "...",
    "upsc_topic": "...",
    "question_category": "..."
  }},
  ...
]
No explanation, no markdown, just the JSON array.
"""


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_untagged(conn, retag=False):
    if retag:
        return conn.execute("SELECT id, question, source_file, source_type FROM questions").fetchall()
    return conn.execute(
        "SELECT id, question, source_file, source_type FROM questions WHERE upsc_subject IS NULL"
    ).fetchall()


def tag_batch(client, batch):
    payload = [
        {
            "id": r["id"],
            "question": r["question"][:600],   # truncate to save tokens
            "source": r["source_file"] or "",
            "source_type": r["source_type"] or "",
        }
        for r in batch
    ]

    resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=TAGGER_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Classify these {len(payload)} questions:\n\n{json.dumps(payload, indent=2)}"
        }],
    )

    raw = resp.content[0].text.strip()
    # strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def apply_tags(conn, tags):
    for t in tags:
        conn.execute("""
            UPDATE questions
            SET upsc_subject=?, upsc_topic=?, broad_category=?, question_category=?
            WHERE id=?
        """, (
            t.get("upsc_subject"),
            t.get("upsc_topic"),
            t.get("upsc_category") or t.get("broad_category"),
            t.get("question_category"),
            t["id"],
        ))
    conn.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retag", action="store_true", help="Retag all questions, not just untagged")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    conn = get_db()

    rows = fetch_untagged(conn, retag=args.retag)
    total = len(rows)

    if total == 0:
        print("All questions already tagged.")
        conn.close()
        return

    print(f"Tagging {total} questions in batches of {BATCH} ...")
    done = 0
    errors = 0

    for i in range(0, total, BATCH):
        chunk = rows[i:i + BATCH]
        try:
            tags = tag_batch(client, chunk)
            apply_tags(conn, tags)
            done += len(tags)
            print(f"  {done}/{total}", end="\r", flush=True)
        except Exception as e:
            errors += len(chunk)
            print(f"\n  ERROR on batch {i//BATCH + 1}: {e}")
        time.sleep(0.3)   # gentle rate limiting

    conn.close()
    print(f"\nDone. Tagged {done} questions, {errors} errors.")

    # Summary
    conn2 = get_db()
    breakdown = conn2.execute(
        "SELECT upsc_subject, COUNT(*) as n FROM questions WHERE upsc_subject IS NOT NULL GROUP BY upsc_subject ORDER BY n DESC"
    ).fetchall()
    print("\nSubject distribution:")
    for r in breakdown:
        print(f"  {r['upsc_subject']:<25} {r['n']}")
    conn2.close()


if __name__ == "__main__":
    main()
