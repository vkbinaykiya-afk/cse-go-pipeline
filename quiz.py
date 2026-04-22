# quiz.py
# CSE-GO Pipeline — Terminal Quiz Viewer
#
# Quick way to review and practice questions from any source:
#   • Real UPSC PYQs (2011-2025)
#   • Agent-generated questions from questions/
#
# ── HOW TO RUN ────────────────────────────────────────────────────────────────
#
#   python quiz.py                          # random PYQs
#   python quiz.py --source pyq            # real UPSC PYQs
#   python quiz.py --source generated      # latest agent-generated batch
#   python quiz.py --source pyq --subject "Environment and Ecology"
#   python quiz.py --source pyq --year 2024
#   python quiz.py --source pyq --subject "Polity" --year 2023
#   python quiz.py --count 10              # number of questions (default 5)
#   python quiz.py --review                # show all questions without interactive mode
#
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import glob
import random
import argparse


PYQ_JSON    = "./pyq_questions.json"
QUESTIONS_DIR = "./questions"

# ANSI colours
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


# ── LOADERS ───────────────────────────────────────────────────────────────────

def load_pyq(subject=None, year=None):
    if not os.path.exists(PYQ_JSON):
        print(f"ERROR: {PYQ_JSON} not found. Run parse_pyq.py first.")
        sys.exit(1)

    with open(PYQ_JSON, encoding="utf-8") as f:
        data = json.load(f)

    # Filter
    if subject:
        data = [q for q in data if (q.get("subject") or "").lower() == subject.lower()]
    if year:
        data = [q for q in data if q.get("year") == int(year)]

    # Only keep records with question + answer + explanation
    data = [q for q in data if q.get("question") and q.get("answer") and q.get("options")]

    return data


def load_generated(batch_file=None):
    if batch_file:
        path = batch_file
    else:
        # Latest agent batch first, then regular batches
        agent  = sorted(glob.glob(os.path.join(QUESTIONS_DIR, "agent_batch_*.json")))
        normal = sorted(glob.glob(os.path.join(QUESTIONS_DIR, "batch_*.json")))
        files  = agent + normal
        if not files:
            print(f"ERROR: No generated batches in {QUESTIONS_DIR}/. Run agent_generate.py first.")
            sys.exit(1)
        path = files[-1]

    print(f"{DIM}Loading: {path}{RESET}\n")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Normalise field names (agent_batch vs regular batch differ slightly)
    normalised = []
    for q in data:
        if not q.get("question") or not q.get("options"):
            continue
        opts = q.get("options", {})
        # Regular batch uses lowercase keys, agent uses uppercase
        if "a" in opts:
            opts = {k.upper(): v for k, v in opts.items()}

        ans = q.get("correct_answer") or q.get("answer", "")
        ans = ans.upper() if ans else ""

        normalised.append({
            "question":      q["question"],
            "options":       opts,
            "answer":        ans,
            "explanation":   q.get("explanation", ""),
            "year":          q.get("year"),
            "subject":       q.get("subject", q.get("topic_query", "")),
            "source_file":   q.get("source_file") or q.get("source", {}).get("filename", ""),
            "source_page":   q.get("source_page") or q.get("source", {}).get("page", ""),
            "question_type": q.get("question_type", ""),
            "difficulty":    q.get("difficulty", ""),
            "cited_extract": q.get("cited_extract", ""),
        })

    return normalised


def normalise_pyq(q):
    """Convert PYQ record to standard display format."""
    opts = q.get("options", {})
    # PYQ options use lowercase keys
    opts_upper = {k.upper(): v for k, v in opts.items()}
    ans = (q.get("answer") or "").upper()
    return {
        "question":      q["question"],
        "options":       opts_upper,
        "answer":        ans,
        "explanation":   q.get("explanation", ""),
        "year":          q.get("year"),
        "subject":       q.get("subject", ""),
        "source_file":   "",
        "source_page":   q.get("source_page", ""),
        "question_type": "",
        "difficulty":    "",
        "cited_extract": "",
    }


# ── DISPLAY ───────────────────────────────────────────────────────────────────

def clear_line():
    print("\033[A\033[K", end="")

def print_header(current, total, q):
    year_str  = f" {q['year']}" if q.get("year") else ""
    subj_str  = f" • {q['subject']}" if q.get("subject") else ""
    diff_str  = f" • {q['difficulty']}" if q.get("difficulty") else ""
    qtype_str = f" [{q['question_type']}]" if q.get("question_type") else ""
    print(f"\n{BOLD}{'═'*62}{RESET}")
    print(f"{BOLD}  Q{current}/{total}{RESET}{CYAN}{year_str}{subj_str}{diff_str}{qtype_str}{RESET}")
    print(f"{'─'*62}")


def print_question(q):
    print(f"\n  {q['question']}\n")
    for key in ["A","B","C","D"]:
        val = q["options"].get(key, "")
        if val:
            print(f"  {BOLD}{key}{RESET})  {val}")


def print_answer(q, user_answer=None):
    correct = q["answer"].upper()
    print()
    if user_answer:
        if user_answer == correct:
            print(f"  {GREEN}{BOLD}✓ Correct!{RESET}")
        else:
            print(f"  {RED}{BOLD}✗ Wrong.{RESET}  You chose {user_answer}.")

    print(f"\n  {GREEN}{BOLD}Answer: ({correct})  {q['options'].get(correct,'')}{RESET}")

    if q.get("explanation"):
        print(f"\n  {BOLD}Explanation:{RESET}")
        # Word-wrap explanation at 70 chars
        words = q["explanation"].split()
        line  = "  "
        for word in words:
            if len(line) + len(word) + 1 > 72:
                print(line)
                line = "  " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)

    extracts = q.get("cited_extracts") or ([q["cited_extract"]] if q.get("cited_extract") else [])
    if extracts:
        for ex in extracts[:3]:
            print(f"\n  {DIM}Extract: \"{str(ex)[:100]}…\"{RESET}")
    elif q.get("source_file"):
        print(f"\n  {DIM}Source: {q['source_file']}  p.{q['source_page']}{RESET}")


# ── QUIZ MODES ────────────────────────────────────────────────────────────────

def interactive_quiz(questions, count):
    """Full interactive quiz — user types answer, sees result."""
    random.shuffle(questions)
    subset = questions[:count]

    score  = 0
    total  = len(subset)
    skipped = 0

    for i, raw_q in enumerate(subset):
        q = normalise_pyq(raw_q) if "answer" in raw_q and len(raw_q.get("answer","")) == 1 and "source_file" not in raw_q else raw_q

        print_header(i + 1, total, q)
        print_question(q)

        while True:
            try:
                inp = input(f"\n  Your answer (A/B/C/D) or S to skip, Q to quit: ").strip().upper()
            except (KeyboardInterrupt, EOFError):
                print("\n\nQuiz ended.")
                break

            if inp == "Q":
                print("\nQuitting.")
                _print_score(score, i, skipped)
                return
            if inp == "S":
                skipped += 1
                print_answer(q)
                break
            if inp in ("A","B","C","D"):
                if inp == q["answer"].upper():
                    score += 1
                print_answer(q, inp)
                break
            print(f"  {YELLOW}Enter A, B, C, D, S, or Q.{RESET}", end="")

        input(f"\n  {DIM}[Press Enter for next question]{RESET}")

    _print_score(score, total, skipped)


def review_mode(questions, count):
    """Review mode — show questions + answers without waiting for input."""
    random.shuffle(questions)
    subset = questions[:count]

    for i, raw_q in enumerate(subset):
        q = normalise_pyq(raw_q) if "source_file" not in raw_q else raw_q
        print_header(i + 1, len(subset), q)
        print_question(q)
        print_answer(q)
        print()

    print(f"\n{BOLD}{'═'*62}{RESET}")
    print(f"  Showed {len(subset)} questions.")
    print(f"{'═'*62}\n")


def _print_score(score, attempted, skipped):
    print(f"\n{BOLD}{'═'*62}{RESET}")
    print(f"  SCORE: {GREEN}{score}{RESET} / {attempted - skipped} attempted  ({skipped} skipped)")
    pct = int(score / (attempted - skipped) * 100) if attempted > skipped else 0
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"  [{bar}] {pct}%")
    print(f"{'═'*62}\n")


# ── SUBJECT LIST ──────────────────────────────────────────────────────────────

def list_subjects(data):
    from collections import Counter
    counts = Counter(q.get("subject") or "General" for q in data)
    print(f"\n{BOLD}Available subjects:{RESET}")
    for subj, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cnt:>4}  {subj}")
    print()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CSE-GO Quiz")
    parser.add_argument("--source",  choices=["pyq","generated"], default="pyq",
                        help="Question source (default: pyq)")
    parser.add_argument("--subject", help="Filter by subject (PYQ only)")
    parser.add_argument("--year",    type=int, help="Filter by year (PYQ only)")
    parser.add_argument("--count",   type=int, default=10, help="Number of questions (default 10)")
    parser.add_argument("--review",  action="store_true", help="Show answers immediately (no interaction)")
    parser.add_argument("--list",    action="store_true", help="List available subjects and exit")
    parser.add_argument("--file",    help="Path to a specific generated batch JSON")
    args = parser.parse_args()

    if args.source == "pyq":
        data = load_pyq(subject=args.subject, year=args.year)
        label = "UPSC PYQs"
    else:
        data = load_generated(batch_file=args.file)
        label = "Generated questions"

    if not data:
        print("No questions found matching your filters.")
        sys.exit(0)

    if args.list:
        list_subjects(data)
        return

    print(f"\n{BOLD}CSE-GO Quiz{RESET}  {CYAN}—  {label}  ({len(data)} available){RESET}")
    if args.subject:
        print(f"  Subject: {args.subject}")
    if args.year:
        print(f"  Year: {args.year}")

    count = min(args.count, len(data))

    if args.review:
        review_mode(data, count)
    else:
        interactive_quiz(data, count)


if __name__ == "__main__":
    main()
