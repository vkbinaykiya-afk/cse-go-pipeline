"""
repair.py — Auto-repair flagged questions using fresh ChromaDB evidence.

For each flagged question:
  1. Reads the flag_reason from check.py
  2. Fetches fresh NCERT + CA chunks for the topic
  3. Asks Claude to fix ONLY the broken part (wrong extract, ambiguous statement, AR logic)
  4. Re-runs check.py logic on the repaired question
  5. Marks as 'pass' or leaves as 'flag' if repair didn't help

Usage:
    python3 repair.py                          # repair latest batch
    python3 repair.py --file questions/x.json  # repair specific file
    python3 repair.py --all                    # repair all flagged across all batches
"""

import os
import re
import sys
import json
import glob
import argparse
import chromadb
import anthropic
from pathlib import Path
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_DIR   = "./chroma-db"
NCERT_COL    = "cse_knowledge_base"
CA_COL       = "current_affairs"
EMBED_MODEL  = "all-MiniLM-L6-v2"
REPAIR_MODEL = "claude-sonnet-4-6"
CHECK_MODEL  = "claude-sonnet-4-6"
QUESTIONS_DIR = "./questions"

REPAIR_SYSTEM = """\
You are an expert UPSC question editor. A question has been flagged with a specific problem.
Your job is to fix ONLY the broken part — do not rewrite the whole question.

You will be given:
  - The original question, options, correct answer, and cited_extracts
  - The flag_reason explaining exactly what is wrong
  - Fresh evidence chunks (NCERT and/or CA) to draw better extracts from

Fix the question by:
  1. If an extract is a heading/label → replace it with a verbatim sentence from the fresh chunks
  2. If an extract doesn't match its statement → find a better extract OR remove that statement
  3. If the question is assertion_reason format AND the subject is NOT Science & Technology /
     Biology / Chemistry / Physics → ALWAYS convert to statement_based format, regardless of
     the flag reason. AR is only permitted for hard science topics.
  4. If a factual claim is ungrounded → either ground it from fresh chunks or remove it

RULES:
  - Each statement must have exactly one extract that explicitly confirms or contradicts it
  - Do not add statements you cannot ground from the provided chunks
  - Do not change the core topic or concept being tested
  - Preserve the difficulty level

Return a single JSON object (not an array) with the repaired question:
{
  "question": "...",
  "options": {"A":"...","B":"...","C":"...","D":"..."},
  "correct_answer": "A",
  "explanation": "...",
  "cited_extracts": ["verbatim extract 1", "verbatim extract 2", ...],
  "source_file": "...",
  "source_type": "...",
  "question_type": "statement_based or assertion_reason",
  "repair_note": "one line describing what was fixed"
}

If you cannot fix the question with the available evidence, return:
{"unfixable": true, "reason": "why"}
"""

CHECK_SYSTEM = """\
You are a hostile examiner reviewing UPSC MCQ questions. For the given question, check:
1. GROUNDING: Is every statement/claim grounded by a cited_extract that explicitly confirms or contradicts it?
2. UNAMBIGUOUS: Is exactly one answer option clearly correct? No two options defensible?
3. DISTRACTORS: Are incorrect options plausibly wrong (not trivially obvious)?

Return JSON only:
{
  "grounding_ok": true/false,
  "ambiguity_ok": true/false,
  "distractors_ok": true/false,
  "pass": true/false,
  "reason": "empty if pass, else specific issue"
}
"""


def get_collections():
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ncert_col = client.get_collection(NCERT_COL, embedding_function=ef)
    try:
        ca_col = client.get_collection(CA_COL, embedding_function=ef)
    except Exception:
        ca_col = None
    return ncert_col, ca_col


def fetch_fresh_chunks(topic, ncert_col, ca_col, n=6):
    lines = []
    ncert_res = ncert_col.query(query_texts=[topic], n_results=n)
    lines.append("NCERT EVIDENCE:")
    for i in range(len(ncert_res["documents"][0])):
        src = ncert_res["metadatas"][0][i].get("source", "")
        pg  = ncert_res["metadatas"][0][i].get("page", 0)
        lines.append(f"  [{src} p.{pg}] {ncert_res['documents'][0][i][:600]}")

    if ca_col:
        ca_res = ca_col.query(query_texts=[topic], n_results=3)
        lines.append("\nCURRENT AFFAIRS EVIDENCE:")
        for i in range(len(ca_res["documents"][0])):
            meta = ca_res["metadatas"][0][i]
            lines.append(f"  [{meta.get('topic','')}] {ca_res['documents'][0][i][:500]}")
            try:
                facts = json.loads(meta.get("upsc_facts", "[]"))
                for f in facts[:2]:
                    if isinstance(f, dict):
                        lines.append(f"    • {f.get('fact','')}")
            except Exception:
                pass

    return "\n".join(lines)


def repair_one(client, q, ncert_col, ca_col):
    topic     = q.get("topic_query", q.get("question", "")[:80])
    flag_reason = q.get("flag_reason", "")
    fresh     = fetch_fresh_chunks(topic, ncert_col, ca_col)

    extracts = q.get("cited_extracts") or ([q["cited_extract"]] if q.get("cited_extract") else [])

    user_msg = f"""FLAG REASON:
{flag_reason}

ORIGINAL QUESTION:
{q.get('question','')}

OPTIONS: {json.dumps(q.get('options',{}))}
CORRECT ANSWER: {q.get('correct_answer','')}
CITED EXTRACTS: {json.dumps(extracts, indent=2)}

FRESH EVIDENCE (use these to find better extracts):
{fresh}

Fix only what the flag_reason identifies. Return the repaired question as JSON."""

    resp = client.messages.create(
        model=REPAIR_MODEL,
        max_tokens=1200,
        system=REPAIR_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = resp.content[0].text.strip()
    raw = re.sub(r'^```json\s*|^```\s*|```\s*$', '', raw, flags=re.MULTILINE).strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            result = json.loads(m.group())
        else:
            return None

    return result


def recheck_one(client, q):
    """Re-run adversarial check on a repaired question."""
    extracts = q.get("cited_extracts") or []
    extracts_str = "\n".join(f'{i+1}. "{e}"' for i, e in enumerate(extracts))

    user_msg = f"""QUESTION:
{q.get('question','')}

OPTIONS: {json.dumps(q.get('options',{}))}
CORRECT ANSWER: {q.get('correct_answer','')}
EXPLANATION: {q.get('explanation','')}

CITED EXTRACTS:
{extracts_str}

Check this question and return JSON verdict."""

    resp = client.messages.create(
        model=CHECK_MODEL,
        max_tokens=400,
        system=CHECK_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = resp.content[0].text.strip()
    raw = re.sub(r'^```json\s*|^```\s*|```\s*$', '', raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"pass": False, "reason": "parse error"}


def find_files(args):
    if args.file:
        return [args.file]
    if args.all:
        return sorted(glob.glob(os.path.join(QUESTIONS_DIR, "*.json")))
    # Latest file
    files = sorted(glob.glob(os.path.join(QUESTIONS_DIR, "*.json")))
    return [files[-1]] if files else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Specific JSON file to repair")
    parser.add_argument("--all", action="store_true", help="Repair all batch files")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set")

    client     = anthropic.Anthropic(api_key=api_key)
    ncert_col, ca_col = get_collections()
    files      = find_files(args)

    total_repaired = 0
    total_failed   = 0
    total_unfixable = 0

    for fpath in files:
        with open(fpath) as f:
            questions = json.load(f)

        flagged = [q for q in questions if q.get("status") == "flag"]
        if not flagged:
            print(f"{os.path.basename(fpath)}: no flagged questions, skipping.")
            continue

        print(f"\n{os.path.basename(fpath)}: repairing {len(flagged)} flagged question(s) ...")

        for q in flagged:
            topic = q.get("topic_query", q.get("question","")[:60])
            print(f"  → {topic[:70]}")

            repaired = repair_one(client, q, ncert_col, ca_col)

            if repaired is None:
                print(f"    ✗ repair parse failed")
                total_failed += 1
                continue

            if repaired.get("unfixable"):
                print(f"    ✗ unfixable: {repaired.get('reason','')}")
                q["repair_note"] = f"unfixable: {repaired.get('reason','')}"
                total_unfixable += 1
                continue

            # Apply repair
            for field in ("question","options","correct_answer","explanation",
                          "cited_extracts","source_file","source_type","question_type"):
                if field in repaired:
                    q[field] = repaired[field]
            q["repair_note"] = repaired.get("repair_note", "repaired")

            # Re-check
            verdict = recheck_one(client, q)
            if verdict.get("pass"):
                q["status"]       = "pass"
                q["grounding_ok"] = verdict.get("grounding_ok", True)
                q["ambiguity_ok"] = verdict.get("ambiguity_ok", True)
                q["distractors_ok"] = verdict.get("distractors_ok", True)
                q["flag_reason"]  = ""
                print(f"    ✓ repaired → PASS  ({q['repair_note']})")
                total_repaired += 1
            else:
                q["flag_reason"] = verdict.get("reason", "repair did not resolve issue")
                print(f"    ~ repaired but still flagged: {q['flag_reason'][:120]}")
                total_failed += 1

        # Save updated file
        with open(fpath, "w") as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"\n{'═'*52}")
    print(f"  REPAIR COMPLETE")
    print(f"  Repaired → pass  : {total_repaired}")
    print(f"  Still flagged    : {total_failed}")
    print(f"  Unfixable        : {total_unfixable}")
    print(f"{'═'*52}")
    print("\nNext step → run api.py startup (or restart) to re-import repaired questions.")


if __name__ == "__main__":
    main()
