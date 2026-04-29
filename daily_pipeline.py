"""
daily_pipeline.py — Automated daily question generation pipeline.

Steps:
  1. Plan 15 diverse topics across UPSC subjects (to yield ~10 passing)
  2. Generate batch (agent_generate batch mode)
  3. Check all questions IN PARALLEL
  4. Repair flagged questions
  5. Re-check repaired questions in parallel
  6. Tag new questions (Haiku)
  7. Sync DB + pre-generate today's daily set
  8. Write summary log

Usage:
    python3 daily_pipeline.py                  # full run
    python3 daily_pipeline.py --dry-run        # plan topics only, no generation
    python3 daily_pipeline.py --topics N       # override topic count (default 15)
"""

import os
import sys
import json
import glob
import time
import logging
import argparse
import sqlite3
import uuid
import datetime
import anthropic
import chromadb
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

import agent_generate
import check as checker
import repair as repairer
import tag_questions as tagger
from question_utils import topic_fingerprint, question_fingerprint, extract_entities, pick_diverse_set
from upsc_syllabus import get_syllabus_text

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHROMA_DIR    = "./chroma-db"
QUESTIONS_DIR = Path("./questions")
LOGS_DIR      = Path("./logs")
DB_PATH       = Path("./cse_go.db")
API_BASE      = os.environ.get("RAILWAY_API_URL", "http://localhost:8000")
EMBED_MODEL   = "all-MiniLM-L6-v2"
PLANNER_MODEL = "claude-sonnet-4-6"
TARGET_PASS   = 10    # aim for this many passing questions per day
GENERATE_N    = 15    # topics to generate (expect ~65% pass rate after repair)
CHECK_WORKERS = 3     # parallel checker threads (conservative to avoid 529 overload)
RECENT_TOPIC_LOOKBACK = 7  # days to look back for recently covered topics

PLANNER_SYSTEM = """\
You are a UPSC Civil Services Examination syllabus expert and question paper designer.
Generate a JSON array of specific topic-queries for batch MCQ generation.

Rules:
- Ground every topic in the official UPSC syllabus provided below
- Each query must be specific enough to retrieve focused NCERT/CA chunks — not a subject name, a precise sub-topic
- Distribute across ALL syllabus sections: Prelims GS, Mains GS-I, GS-II, GS-III (not just one area)
- Vary question types: some conceptual, some factual, some current-affairs-linked, some application-based
- Do NOT repeat any topic from the RECENTLY COVERED list
- Avoid topics that are too obscure or too broad
- Return ONLY a valid JSON array of strings, no explanation, no markdown
"""


# ── SETUP ─────────────────────────────────────────────────────────────────────

def setup_logging():
    LOGS_DIR.mkdir(exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"pipeline_{date_str}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return log_file


def get_collections():
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ncert_col = client.get_collection(agent_generate.NCERT_COL, embedding_function=ef)
    pyq_col   = client.get_collection(agent_generate.PYQ_COL,   embedding_function=ef)
    try:
        ca_col = client.get_collection(agent_generate.CA_COL, embedding_function=ef)
    except Exception:
        ca_col = None
    return ncert_col, pyq_col, ca_col


# ── STEP 1: PLAN TOPICS ───────────────────────────────────────────────────────

def get_recent_topics(days=RECENT_TOPIC_LOOKBACK):
    """Fetch topic labels from the last N days of daily sets to avoid repetition."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    recent_sets = conn.execute(
        "SELECT question_ids FROM daily_sets ORDER BY date DESC LIMIT ?", (days,)
    ).fetchall()
    topics = []
    for row in recent_sets:
        ids = json.loads(row["question_ids"])
        for qid in ids:
            q = conn.execute(
                "SELECT topic_query, upsc_topic FROM questions WHERE id=?", (qid,)
            ).fetchone()
            if q:
                label = q["upsc_topic"] or q["topic_query"] or ""
                if label:
                    topics.append(label)
    conn.close()
    return list(dict.fromkeys(topics))  # deduplicated, order preserved


def plan_topics(client, n):
    """Generate n diverse topic queries grounded in the UPSC syllabus, avoiding recent topics."""
    recent = get_recent_topics()
    recent_block = ""
    if recent:
        recent_block = "\nRECENTLY COVERED (do not repeat these):\n" + \
                       "\n".join(f"- {t}" for t in recent[:30]) + "\n"

    syllabus = get_syllabus_text()

    resp = client.messages.create(
        model=PLANNER_MODEL,
        max_tokens=1500,
        system=PLANNER_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"Generate {n} specific topic-queries for UPSC MCQ generation today.\n"
                f"{recent_block}\n"
                f"UPSC SYLLABUS (ground all topics in this):\n{syllabus}\n\n"
                f"Requirements:\n"
                f"- Spread across Prelims GS, Mains GS-I, GS-II, and GS-III sections\n"
                f"- At least 2 topics from Current Affairs / in-news topics\n"
                f"- At least 2 topics from History or Culture\n"
                f"- At least 2 topics from Polity or Governance\n"
                f"- At least 2 topics from Economy or Science & Technology\n"
                f"- At least 1 topic from Geography or Environment\n"
                f"- Remaining: any underrepresented syllabus area\n"
                f"- Each query should be a specific sub-topic, NOT a broad subject name\n\n"
                f"Return a JSON array of exactly {n} strings."
            )
        }]
    )
    import re
    raw = resp.content[0].text.strip()
    raw = re.sub(r'^```json\s*|^```\s*|```\s*$', '', raw, flags=re.MULTILINE).strip()
    try:
        topics = json.loads(raw)
        return topics[:n]
    except Exception:
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            return json.loads(m.group())[:n]
        return []



# ── STEP 2: GENERATE ─────────────────────────────────────────────────────────

def generate(client, topics, ncert_col, pyq_col, ca_col):
    logging.info(f"Generating {len(topics)} questions in batch mode ...")
    retrieved = agent_generate.batch_retrieve(topics, ncert_col, ca_col)

    sonnet_items = [r for r in retrieved if r["tier"] == "sonnet"]
    haiku_items  = [r for r in retrieved if r["tier"] == "haiku"]
    logging.info(f"  Sonnet tier: {len(sonnet_items)} | Haiku tier: {len(haiku_items)}")

    records = []
    if haiku_items:
        qs = agent_generate.batch_generate(client, haiku_items, "haiku")
        records.extend(qs)
        logging.info(f"  Haiku generated: {len(qs)} (after self-verify)")

    if sonnet_items:
        qs = agent_generate.batch_generate(client, sonnet_items, "sonnet")
        records.extend(qs)
        logging.info(f"  Sonnet generated: {len(qs)} (after self-verify)")

    # Save to file
    output_file, saved = agent_generate.save_batch(records, str(QUESTIONS_DIR))
    logging.info(f"  Saved {len(saved)} questions → {output_file}")
    return output_file, saved


# ── STEP 3: PARALLEL CHECK ────────────────────────────────────────────────────

def check_one(args):
    client, record = args
    if record.get("status") != "pending_check":
        return record
    for attempt in range(4):
        try:
            return checker.check_one_question(client, record)
        except Exception as e:
            if "overloaded" in str(e).lower() or "529" in str(e):
                wait = 10 * (2 ** attempt)
                logging.warning(f"  API overloaded, retrying in {wait}s ...")
                time.sleep(wait)
            else:
                logging.warning(f"  Check error: {e}")
                break
    record["status"] = "flag"
    record["flag_reason"] = "check failed after retries"
    return record


def check_parallel(records, workers=CHECK_WORKERS):
    """Check all pending questions in parallel."""
    api_key = os.environ["ANTHROPIC_API_KEY"]
    pending = [r for r in records if r.get("status") == "pending_check"]
    logging.info(f"Checking {len(pending)} questions in parallel (workers={workers}) ...")

    # Each thread needs its own client instance
    args = [(anthropic.Anthropic(api_key=api_key), r) for r in pending]

    results_map = {r["topic_query"]: r for r in records}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(check_one, a): a[1] for a in args}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            tq = result.get("topic_query", "")
            # Update in records list by matching topic_query
            for r in records:
                if r.get("topic_query") == tq and r.get("status") == "pending_check":
                    r.update(result)
                    break
            done += 1
            status = "✓" if result.get("status") == "pass" else "✗"
            logging.info(f"  [{done}/{len(pending)}] {status} {tq[:55]}")

    passed  = sum(1 for r in records if r.get("status") == "pass")
    flagged = sum(1 for r in records if r.get("status") == "flag")
    logging.info(f"  Check complete: {passed} pass, {flagged} flag")
    return records


# ── STEP 4 & 5: REPAIR + RECHECK ─────────────────────────────────────────────

def repair_and_recheck(records, ncert_col, ca_col):
    api_key = os.environ["ANTHROPIC_API_KEY"]
    client  = anthropic.Anthropic(api_key=api_key)

    flagged = [r for r in records if r.get("status") == "flag"]
    if not flagged:
        logging.info("No flagged questions to repair.")
        return records

    logging.info(f"Repairing {len(flagged)} flagged questions ...")
    repaired_count = 0

    for r in records:
        if r.get("status") != "flag":
            continue
        repaired = repairer.repair_one(client, r, ncert_col, ca_col)
        if repaired is None or repaired.get("unfixable"):
            logging.info(f"  ✗ unfixable: {r.get('topic_query','')[:50]}")
            continue

        for field in ("question","options","correct_answer","explanation",
                      "cited_extracts","source_file","source_type","question_type"):
            if field in repaired:
                r[field] = repaired[field]
        r["repair_note"] = repaired.get("repair_note", "repaired")
        r["status"] = "pending_check"
        repaired_count += 1

    logging.info(f"  Repaired {repaired_count} questions, rechecking in parallel ...")
    records = check_parallel(records)
    return records


# ── STEP 6: TAG ───────────────────────────────────────────────────────────────

def tag_new_questions():
    api_key = os.environ["ANTHROPIC_API_KEY"]
    client  = anthropic.Anthropic(api_key=api_key)
    conn    = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT id, question, source_file, source_type FROM questions WHERE upsc_subject IS NULL"
    ).fetchall()

    if not rows:
        logging.info("All questions already tagged.")
        conn.close()
        return

    logging.info(f"Tagging {len(rows)} new questions ...")
    for i in range(0, len(rows), tagger.BATCH):
        chunk = rows[i:i + tagger.BATCH]
        try:
            tags = tagger.tag_batch(client, chunk)
            tagger.apply_tags(conn, tags)
        except Exception as e:
            logging.warning(f"  Tag batch error: {e}")
        time.sleep(0.3)
    conn.close()
    logging.info("  Tagging complete.")


# ── STEP 7: SYNC DB ───────────────────────────────────────────────────────────

def sync_db():
    """Sync statuses to DB and pre-generate today's daily set."""
    try:
        resp = requests.post(f"{API_BASE}/internal/sync", timeout=10)
        logging.info(f"  DB sync via API: {resp.status_code}")
    except Exception:
        # API not running — sync directly
        logging.info("  API not reachable, syncing DB directly ...")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        # Ensure repair_note column exists (may be missing in older DBs)
        conn.execute("ALTER TABLE questions ADD COLUMN repair_note TEXT")
        conn.commit()
        files = glob.glob(str(QUESTIONS_DIR / "*.json"))
        updated = 0
        for fpath in files:
            with open(fpath) as f:
                try:
                    data = json.load(f)
                except Exception:
                    continue
            for q in data:
                q_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, q.get("question","") + q.get("source_file","")))
                status = q.get("status")
                if status in ("pass", "flag"):
                    conn.execute(
                        "UPDATE questions SET status=?, flag_reason=?, repair_note=? WHERE id=?",
                        (status, q.get("flag_reason"), q.get("repair_note"), q_id)
                    )
                    updated += conn.execute("SELECT changes()").fetchone()[0]
        conn.commit()
        conn.close()
        logging.info(f"  Direct sync: {updated} records updated.")

    # Pre-generate today's daily set by calling the API
    today = datetime.date.today().isoformat()
    try:
        resp = requests.get(f"{API_BASE}/daily", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            logging.info(f"  Daily set for {today}: {data['summary']['total']} questions ready.")
        else:
            logging.warning(f"  Daily set call returned {resp.status_code}")
    except Exception as e:
        logging.warning(f"  Could not pre-generate daily set: {e}")


# ── STEP 7: BUILD DAILY SET ───────────────────────────────────────────────────

def build_daily_set(records, date_str):
    """
    From today's checked+repaired records, select today's 10-question set.

    Filters applied (in order):
      1. Quality: status == 'pass' only
      2. Last-30-days topic exclusion: skip any topic already in a recent daily set
      3. Within-set dedup: topic fingerprint + question text + named entity
      4. If >10 valid: take first 10, log extras as pool candidates
      5. If <10: top up with PYQ questions (unique to the remaining set)

    Writes the final set to daily_sets table.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    today = date_str

    # ── Get recent topic fingerprints (last 30 daily sets) ────────────────────
    recent_sets = conn.execute(
        "SELECT question_ids FROM daily_sets ORDER BY date DESC LIMIT 30"
    ).fetchall()

    recently_used_ids = set()
    recent_topic_fps  = set()
    for row in recent_sets:
        ids = json.loads(row["question_ids"])
        recently_used_ids.update(ids)
        for qid in ids:
            q = conn.execute(
                "SELECT COALESCE(upsc_topic, topic_query, source_file) AS topic_key "
                "FROM questions WHERE id=?", (qid,)
            ).fetchone()
            if q:
                fp = topic_fingerprint(q["topic_key"] or "")
                if fp:
                    recent_topic_fps.add(fp)

    # ── Build candidate list from today's passing records ─────────────────────
    passing = [r for r in records if r.get("status") == "pass"]
    candidates = []
    for r in passing:
        q_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, r.get("question", "") + r.get("source_file", "")))
        if q_id in recently_used_ids:
            continue
        topic_key = r.get("upsc_topic") or r.get("topic_query") or r.get("source_file", "")
        candidates.append({"id": q_id, "topic_key": topic_key, "question": r.get("question", "")})

    logging.info(f"  {len(passing)} passing today → {len(candidates)} after 30-day topic filter")

    # ── Select diverse set of up to 10 ────────────────────────────────────────
    selected, seen_t, seen_q, seen_ents = pick_diverse_set(
        candidates, limit=10, recent_topic_fps=recent_topic_fps
    )

    # Log extras as pool candidates
    extras = [c for c in candidates if c["id"] not in set(selected)]
    if extras:
        logging.info(f"  {len(extras)} extra valid question(s) available for pool (roadmap):")
        for e in extras:
            logging.info(f"    • {e['topic_key'][:60]}")

    # ── PYQ top-up if short ───────────────────────────────────────────────────
    if len(selected) < 10:
        needed = 10 - len(selected)
        logging.info(f"  Short by {needed}, topping up with PYQ questions ...")

        # PYQs use a shorter 7-day exclusion window — they're evergreen by nature
        pyq_recent_sets = conn.execute(
            "SELECT question_ids FROM daily_sets ORDER BY date DESC LIMIT 7"
        ).fetchall()
        pyq_recently_used = set()
        for row in pyq_recent_sets:
            pyq_recently_used.update(json.loads(row["question_ids"]))
        pyq_recently_used.update(set(selected))

        excl_pyq = list(pyq_recently_used)
        pyqs = conn.execute(
            "SELECT id, question, COALESCE(upsc_topic, topic_query, source_file) AS topic_key "
            "FROM questions "
            "WHERE status='pass' "
            "AND (source_type='pyq' OR LOWER(source_file) LIKE '%pyq%') "
            + (f"AND id NOT IN ({','.join('?' * len(excl_pyq))})" if excl_pyq else ""),
            excl_pyq
        ).fetchall()

        pyq_candidates = [{"id": r["id"], "topic_key": r["topic_key"], "question": r["question"]}
                          for r in pyqs]
        pyq_ids, _, pyq_seen_q, _ = pick_diverse_set(
            pyq_candidates, limit=needed,
            recent_topic_fps=seen_t,
            strict_entity=False
        )
        seen_q.update(pyq_seen_q)
        selected.extend(pyq_ids)
        logging.info(f"  Added {len(pyq_ids)} PYQ question(s). Total: {len(selected)}")

    # ── Final fallback: any passing NCERT question not used recently ──────────
    if len(selected) < 10:
        needed = 10 - len(selected)
        logging.info(f"  Still short by {needed}, falling back to NCERT pool ...")

        excl_all = list(recently_used_ids | set(selected))
        ncert_rows = conn.execute(
            "SELECT id, question, COALESCE(upsc_topic, topic_query, source_file) AS topic_key "
            "FROM questions WHERE status='pass' "
            "AND (source_type='ncert' OR source_type IS NULL OR source_type='') "
            + (f"AND id NOT IN ({','.join('?' * len(excl_all))})" if excl_all else "")
            + " ORDER BY RANDOM()",
            excl_all
        ).fetchall()
        ncert_cands = [{"id": r["id"], "topic_key": r["topic_key"], "question": r["question"]}
                       for r in ncert_rows]
        ncert_ids, _, _, _ = pick_diverse_set(
            ncert_cands, limit=needed, recent_topic_fps=seen_t, strict_entity=False
        )
        selected.extend(ncert_ids)
        logging.info(f"  Added {len(ncert_ids)} NCERT question(s). Total: {len(selected)}")

    # ── Write to DB ───────────────────────────────────────────────────────────
    conn.execute(
        "INSERT OR REPLACE INTO daily_sets (date, question_ids, created_at) VALUES (?,?,?)",
        (today, json.dumps(selected), datetime.datetime.now(datetime.timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()

    logging.info(f"  Daily set locked: {len(selected)} questions for {today}")

    # Push to Railway API if configured
    railway_url = os.environ.get("RAILWAY_API_URL", "")
    pipeline_secret = os.environ.get("PIPELINE_SECRET", "")
    if railway_url:
        try:
            # Build a lookup from today's in-memory records so newly generated
            # questions (not yet in SQLite) are still included in the payload.
            records_by_id = {}
            for r in records:
                r_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, r.get("question", "") + r.get("source_file", "")))
                records_by_id[r_id] = r

            conn2 = sqlite3.connect(DB_PATH)
            conn2.row_factory = sqlite3.Row
            q_payload = []
            for q_id in selected:
                r = records_by_id.get(q_id)
                if r:
                    # Use in-memory record for today's new questions
                    q_payload.append({
                        "id": q_id,
                        "question": r.get("question", ""),
                        "options": r.get("options", {}),
                        "correct_answer": r.get("correct_answer", ""),
                        "explanation": r.get("explanation", ""),
                        "subject": r.get("subject", ""),
                        "difficulty": r.get("difficulty", ""),
                        "question_type": r.get("question_type", ""),
                        "source_type": r.get("source_type", ""),
                        "source_file": r.get("source_file", ""),
                        "source_page": r.get("source_page"),
                        "status": r.get("status", "pass"),
                        "cited_extracts": r.get("cited_extracts", []),
                        "upsc_subject": r.get("upsc_subject"),
                        "upsc_topic": r.get("upsc_topic"),
                        "broad_category": r.get("broad_category"),
                        "question_category": r.get("question_category"),
                    })
                else:
                    # Fall back to SQLite for historical (PYQ/pool) questions
                    row = conn2.execute("SELECT * FROM questions WHERE id=?", (q_id,)).fetchone()
                    if row:
                        q_payload.append({
                            "id": row["id"], "question": row["question"],
                            "options": json.loads(row["options"]) if row["options"] else {},
                            "correct_answer": row["correct"], "explanation": row["explanation"],
                            "subject": row["subject"], "difficulty": row["difficulty"],
                            "question_type": row["question_type"], "source_type": row["source_type"],
                            "source_file": row["source_file"] or "", "source_page": row["source_page"],
                            "status": row["status"], "cited_extracts": json.loads(row["extracts"] or "[]"),
                            "upsc_subject": row["upsc_subject"], "upsc_topic": row["upsc_topic"],
                            "broad_category": row["broad_category"], "question_category": row["question_category"],
                        })
                    else:
                        logging.warning(f"  Question {q_id} not found in records or SQLite — skipping from payload")
            conn2.close()
            if q_payload:
                qresp = requests.post(
                    f"{railway_url}/internal/push-questions",
                    json={"questions": q_payload, "secret": pipeline_secret},
                    timeout=30,
                )
                if qresp.ok:
                    logging.info(f"  Pushed {len(q_payload)} questions to Railway ({qresp.json()})")
                else:
                    logging.warning(f"  Railway questions push failed: {qresp.status_code} {qresp.text[:120]}")

            resp = requests.post(
                f"{railway_url}/internal/push-daily-set",
                json={"date": today, "question_ids": selected, "secret": pipeline_secret},
                timeout=15,
            )
            if resp.ok:
                logging.info(f"  Pushed daily set to Railway ({resp.json()})")
            else:
                logging.warning(f"  Railway push failed: {resp.status_code} {resp.text[:120]}")
        except Exception as e:
            logging.warning(f"  Railway push error (non-fatal): {e}")

    return selected


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true", help="Plan topics only, no generation")
    parser.add_argument("--topics",   type=int, default=GENERATE_N, help="Number of topics to generate")
    args = parser.parse_args()

    log_file = setup_logging()
    logging.info("=" * 60)
    logging.info("CSE-GO DAILY PIPELINE")
    logging.info(f"Target: {TARGET_PASS} passing questions")
    logging.info("=" * 60)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        logging.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.Anthropic()

    # Collections
    ncert_col, pyq_col, ca_col = get_collections()
    logging.info(f"NCERT: {ncert_col.count()} | CA: {ca_col.count() if ca_col else 0}")

    # Step 1: Plan
    logging.info(f"\n[1/7] Planning {args.topics} topics ...")
    topics = plan_topics(client, args.topics)
    logging.info(f"  Topics planned:")
    for t in topics:
        logging.info(f"    • {t}")

    if args.dry_run:
        logging.info("Dry run complete.")
        return

    # Step 2: Generate
    logging.info(f"\n[2/7] Generating ...")
    output_file, records = generate(client, topics, ncert_col, pyq_col, ca_col)

    if not records:
        logging.error("No questions generated. Exiting.")
        sys.exit(1)

    # Step 3: Check in parallel
    logging.info(f"\n[3/7] Checking {len(records)} questions in parallel ...")
    records = check_parallel(records)

    # Save after check
    with open(output_file, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    pass_count = sum(1 for r in records if r.get("status") == "pass")
    logging.info(f"  After check: {pass_count}/{len(records)} passing")

    # Step 4-5: Repair + recheck
    logging.info(f"\n[4/7] Repairing flagged questions ...")
    records = repair_and_recheck(records, ncert_col, ca_col)

    # Save after repair
    with open(output_file, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    pass_count = sum(1 for r in records if r.get("status") == "pass")
    logging.info(f"  After repair: {pass_count}/{len(records)} passing")

    # Step 5: Sync DB so new questions are importable by tagger + set builder
    logging.info(f"\n[5/7] Syncing DB ...")
    sync_db()

    # Step 6: Tag new questions
    logging.info(f"\n[6/7] Tagging new questions ...")
    tag_new_questions()

    # Step 7: Build today's daily set
    today = datetime.date.today().isoformat()
    logging.info(f"\n[7/7] Building daily set for {today} ...")
    selected_ids = build_daily_set(records, today)

    # Summary
    logging.info(f"\n{'='*60}")
    logging.info("PIPELINE COMPLETE")
    logging.info(f"  Topics planned    : {len(topics)}")
    logging.info(f"  Questions generated: {len(records)}")
    logging.info(f"  Passing           : {pass_count}")
    logging.info(f"  Today's set       : {len(selected_ids)} questions")
    logging.info(f"  Log               : {log_file}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
