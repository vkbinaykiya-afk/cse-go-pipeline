# parse_pyq.py
# CSE-GO Pipeline — PYQ Parser
#
# Parses "Prelims PYQ (2011-2025).pdf" (Physics Wallah / PW OnlyIAS edition)
# into a structured JSON file: ./pyq_questions.json
#
# Handles two formats inside the PDF:
#   A. Sequential papers (2025, 2024, 2023): questions then explanations, full papers
#   B. Subject-wise sections (2011-2022): questions grouped by topic with year tags
#
# Output per question:
#   {
#     "year":        2025,
#     "subject":     "Science and Technology",   (None for 2023-2025)
#     "q_num":       7,
#     "question":    "Consider the following statements ...",
#     "options":     {"a": "...", "b": "...", "c": "...", "d": "..."},
#     "answer":      "c",
#     "explanation": "Statement I is correct: ...",
#     "source_page": 23    (PDF page number, 1-indexed)
#   }
#
# ── HOW TO RUN ────────────────────────────────────────────────────────────────
#   python parse_pyq.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import json
import fitz   # PyMuPDF

PDF_PATH   = "./source-docs/Prelims PYQ (2011-2025).pdf"
OUTPUT     = "./pyq_questions.json"


# ── HELPERS ───────────────────────────────────────────────────────────────────

def clean(text):
    """Collapse extra whitespace but keep newlines meaningful."""
    # Replace tab + multiple spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Collapse 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_options(text):
    """
    Pull (a)/(b)/(c)/(d) options from a question block.
    Returns dict {a: ..., b: ..., c: ..., d: ...} or None if not all four found.
    """
    opts = {}
    # Match (a)  option text   up to next option or end
    pattern = re.compile(
        r'\(([abcd])\)\s+(.*?)(?=\s*\([abcd]\)\s+|\Z)',
        re.DOTALL | re.IGNORECASE
    )
    for m in pattern.finditer(text):
        letter = m.group(1).lower()
        value  = clean(m.group(2))
        opts[letter] = value

    if len(opts) == 4:
        return opts
    return None


def strip_header_noise(text):
    """Remove running headers like 'Prelims 2025 Question Paper\n3\n' and 'UPSC Prelims PYQs\n14\n'."""
    text = re.sub(r'Prelims \d{4} Question Paper\s*\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'UPSC Prelims PYQs\s*\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'Ancient and Medieval History\s*\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'Modern History\s*\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'Art and Culture\s*\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'Polity\s*\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'Indian Economy\s*\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'Environment and Ecology\s*\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'Geography\s*\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'Science (?:and|&) Technology\s*\n\s*\d+\s*\n', '\n', text)
    return text


# ── QUESTION PARSER ───────────────────────────────────────────────────────────

# Matches the start of a numbered question:   "  7.  question text..."
# Tabs and spaces vary, so we match any leading whitespace + number + dot
Q_START = re.compile(r'(?:^|\n)\s{0,4}(\d{1,3})\.\s+(?!\()', re.MULTILINE)

def parse_questions(text, year, subject, page_hint, strict_monotonic=False):
    """
    Given a block of question text, returns a list of partially-filled records.
    Each record has: year, subject, q_num, question, options, source_page.
    answer and explanation are filled later when we parse explanations.

    strict_monotonic=True: only accept question numbers that are greater than
    the last accepted number, suppressing sub-list items (1,2,3 inside a question).
    """
    text = strip_header_noise(text)
    records = []

    # Find all candidate question start positions
    starts = [(m.start(), int(m.group(1))) for m in Q_START.finditer(text)]

    last_q_num = 0

    for idx, (pos, q_num) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(text)
        block = text[pos:end]

        opts = extract_options(block)
        if opts is None:
            continue  # no 4 options = not a well-formed question

        # Skip blocks whose stem (text before first option) is too short to be a
        # real question — list items like "3. Fuel Cell electric hybrid vehicles"
        # have only 3-5 words; genuine question stems have 8+.
        first_opt_check = re.search(r'\([abcd]\)', block, re.IGNORECASE)
        stem_check = block[:first_opt_check.start()].strip() if first_opt_check else block.strip()
        if len(stem_check.split()) < 8:
            continue

        last_q_num = q_num

        # Question text = everything before the first option
        first_opt = re.search(r'\([abcd]\)', block, re.IGNORECASE)
        q_text = block[:first_opt.start()].strip() if first_opt else block.strip()

        # Remove the leading "N. " from the question text
        q_text = re.sub(r'^\s*\d+\.\s+', '', q_text).strip()

        # Extract year tag if present — check both the raw block and the q_text
        # because year tag may appear anywhere: after question stem, in the block header
        year_tag = re.search(r'\((20\d{2})\)', block) or re.search(r'\((20\d{2})\)', q_text)
        actual_year = int(year_tag.group(1)) if year_tag else year
        q_text = re.sub(r'\(20\d{2}\)', '', q_text).strip()

        records.append({
            "year":        actual_year,
            "subject":     subject,
            "q_num":       q_num,
            "question":    clean(q_text),
            "options":     opts,
            "answer":      None,
            "explanation": None,
            "source_page": page_hint,
        })

    return records


# ── EXPLANATION PARSER ────────────────────────────────────────────────────────

# Matches "7.  (c)  explanation text..."  at the start of an explanation block
EXP_START = re.compile(
    r'(?:^|\n)\s{0,4}(\d{1,3})\.\s+\(([a-d/]+)\)\s+',
    re.MULTILINE | re.IGNORECASE
)

# Strip PW hint boxes — we keep only the core explanation
PW_HINT_RE = re.compile(
    r'PW ONLYIAS (?:SUPER HINT|EXTRA EDGE)[^\n]*\n.*?(?=\n\s{0,4}\d+\.\s+\([a-d]\)|\Z)',
    re.DOTALL | re.IGNORECASE
)

def parse_explanations(text):
    """
    Given explanation text, returns dict {q_num: (answer_letter, explanation_text)}.
    """
    text = strip_header_noise(text)
    # Remove PW hint boxes to keep explanations clean
    text = PW_HINT_RE.sub('', text)

    result = {}
    starts = [(m.start(), int(m.group(1)), m.group(2).lower()) for m in EXP_START.finditer(text)]

    for idx, (pos, q_num, answer) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(text)
        # The explanation starts after the "N. (letter) " header
        m = EXP_START.search(text[pos:pos + 40])
        exp_start = pos + m.end() if m else pos
        exp_text = clean(text[exp_start:end])

        # Use first letter only if answer is like "c/d"
        answer_clean = answer.split('/')[0].strip()

        result[q_num] = (answer_clean, exp_text)

    return result


# ── COMPACT ANSWER KEY PARSER ─────────────────────────────────────────────────

COMPACT_KEY_RE = re.compile(r'(\d{1,3})\.\s+\(([a-d/]+)\)', re.IGNORECASE)

def parse_compact_answer_key(text):
    """
    Parse the compact answer key page: "1. (c)  2. (d)  3. (c) ..."
    Returns dict {q_num: answer_letter}.
    """
    result = {}
    for m in COMPACT_KEY_RE.finditer(text):
        q_num  = int(m.group(1))
        answer = m.group(2).lower().split('/')[0].strip()
        result[q_num] = answer
    return result


# ── SECTION SCANNER ───────────────────────────────────────────────────────────

def get_pages_text(doc, start_idx, end_idx):
    """Concatenate text from PDF pages [start_idx, end_idx) (0-indexed)."""
    parts = []
    for i in range(start_idx, min(end_idx, len(doc))):
        parts.append(doc[i].get_text())
    return '\n'.join(parts)


def find_section_boundaries(doc):
    """
    Scan the PDF and return structured section info:
    [
      {"type": "sequential", "year": 2025, "start": 7, "end": 51},
      {"type": "subjectwise", "subject": "Ancient and Medieval History", "start": 121, "end": 141},
      ...
    ]
    """
    sections = []
    n = len(doc)

    # Sequential year papers
    YEAR_HEADERS = {
        2025: "Prelims 2025 Question Paper",
        2024: "Prelims 2024 Question Paper",
        2023: "Prelims 2023 Question Paper",
    }

    # Subject-wise section headers (2011-2022)
    SUBJECT_HEADERS = [
        "PREVIOUS YEARS QUESTIONS – PRELIMS 2011 TO 2022",
        "Ancient and Medieval History",
        "Modern History",
        "Art and Culture",
        "Polity",
        "Indian Economy",
        "Environment and Ecology",
        "Geography",
        "Science",
    ]

    year_starts = {}
    subject_starts = {}

    for i in range(n):
        text = doc[i].get_text()
        first300 = text[:300]
        # Skip TOC pages (contain long dotted lines like ".........")
        is_toc = '........' in first300 or 'Contents' in first300
        for year, header in YEAR_HEADERS.items():
            if header in first300 and year not in year_starts and not is_toc:
                year_starts[year] = i

    # Determine end of each year section (start of next year or end of file)
    sorted_years = sorted(year_starts.items(), key=lambda x: x[1])
    for idx, (year, start) in enumerate(sorted_years):
        end = sorted_years[idx + 1][1] if idx + 1 < len(sorted_years) else None
        sections.append({
            "type":    "sequential",
            "year":    year,
            "subject": None,
            "start":   start,
            "end":     end
        })

    # Subject-wise sections start after sequential papers
    last_seq_end = max(s["end"] for s in sections if s["end"]) if sections else 0
    # subject_order normalised names — detection uses aliases below
    subject_order = [
        "Ancient and Medieval History",
        "Modern History",
        "Art and Culture",
        "Polity",
        "Indian Economy",
        "Environment and Ecology",
        "Geography",
        "Science and Technology",
    ]
    # Allow alternate spellings in the PDF
    SUBJECT_ALIASES = {
        "Ancient and Medieval History": ["Ancient and Medieval History"],
        "Modern History":               ["Modern History"],
        "Art and Culture":              ["Art and Culture", "Art & Culture"],
        "Polity":                       ["Polity"],
        "Indian Economy":               ["Indian Economy"],
        "Environment and Ecology":      ["Environment and Ecology"],
        "Geography":                    ["Geography"],
        "Science and Technology":       ["Science and Technology", "Science & Technology"],
    }

    subj_pages = {}
    for i in range(last_seq_end, n):
        full_text = doc[i].get_text()
        is_section_header = "PREVIOUS YEARS" in full_text or "PYQs Analysis" in full_text
        if not is_section_header:
            continue
        for subj, aliases in SUBJECT_ALIASES.items():
            if subj not in subj_pages:
                if any(a in full_text for a in aliases):
                    subj_pages[subj] = i

    sorted_subj = sorted(subj_pages.items(), key=lambda x: x[1])
    for idx, (subj, start) in enumerate(sorted_subj):
        end = sorted_subj[idx + 1][1] if idx + 1 < len(sorted_subj) else n
        sections.append({
            "type":    "subjectwise",
            "year":    None,
            "subject": subj,
            "start":   start,
            "end":     end
        })

    return sections


# ── SEQUENTIAL SECTION PROCESSOR ─────────────────────────────────────────────

def process_sequential(doc, section):
    """Process a full year paper (2023/2024/2025): find answer key, parse Q+A."""
    year    = section["year"]
    start   = section["start"]
    end     = section["end"] or len(doc)
    records = []

    # Find the answer key page within this section
    ak_page = None
    for i in range(start, end):
        text = doc[i].get_text()
        if 'Answer Key' in text:
            ak_page = i
            break

    if ak_page is None:
        # No explicit answer key page — explanations start where we first see "1. (letter) text"
        # Find the boundary: first page where we see N. (letter) pattern at the start
        for i in range(start, end):
            text = doc[i].get_text()
            if re.search(r'^\s{0,4}1\.\s+\([a-d]\)\s+\w', text, re.MULTILINE):
                ak_page = i
                break

    if ak_page is None:
        print(f"  WARNING: Could not find answer/explanation boundary for {year}")
        return []

    # Question pages = start → ak_page (exclusive)
    q_text = get_pages_text(doc, start, ak_page)

    # Parse compact answer key if present on ak_page, otherwise extract from explanations
    ak_text   = doc[ak_page].get_text()
    ans_key   = parse_compact_answer_key(ak_text) if 'Answer Key' in ak_text else {}

    # Explanation pages = ak_page → end
    exp_text  = get_pages_text(doc, ak_page, end)
    exp_map   = parse_explanations(exp_text)

    # Merge answer key with explanation-derived answers (exp_map wins on conflict)
    for q_num, (letter, _) in exp_map.items():
        ans_key[q_num] = letter

    qs = parse_questions(q_text, year, None, start + 1)

    # Attach answers and explanations
    for q in qs:
        n = q["q_num"]
        q["answer"]      = ans_key.get(n)
        q["explanation"] = exp_map.get(n, (None, None))[1]

    records.extend(qs)
    return records


# ── SUBJECT-WISE SECTION PROCESSOR ───────────────────────────────────────────

def process_subjectwise(doc, section):
    """Process a subject section from the 2011-2022 portion."""
    subj  = section["subject"]
    start = section["start"]
    end   = section["end"] or len(doc)

    all_text = get_pages_text(doc, start, end)

    # Find where explanations start: first "N. (letter) " pattern
    # that is NOT followed by a question-style "(a) ... (b) ... (c) ... (d) ..."
    # Strategy: find first explanation block, use that as split point
    exp_match = EXP_START.search(all_text)
    if exp_match:
        split_pos = exp_match.start()
        # Go back a bit — some explanation pages have a few questions before the first explanation
        # Use the split to separate question text from explanation text
        q_text   = all_text[:split_pos]
        exp_text = all_text[split_pos:]
    else:
        q_text   = all_text
        exp_text = ""

    # Parse questions (year is embedded as (20XX) tag)
    qs      = parse_questions(q_text, None, subj, start + 1)
    exp_map = parse_explanations(exp_text)

    # Build answer key from explanations
    ans_key = {n: letter for n, (letter, _) in exp_map.items()}

    for q in qs:
        n = q["q_num"]
        q["answer"]      = ans_key.get(n)
        q["explanation"] = exp_map.get(n, (None, None))[1]

    return qs


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(PDF_PATH):
        print(f"ERROR: {PDF_PATH} not found.")
        print("Copy the PYQ PDF into ./source-docs/ first.")
        return

    print(f"\nOpening: {PDF_PATH}")
    doc = fitz.open(PDF_PATH)
    print(f"Pages: {len(doc)}")

    print("\nScanning section boundaries ...")
    sections = find_section_boundaries(doc)

    for s in sections:
        if s["type"] == "sequential":
            print(f"  {s['year']} — pages {s['start']+1}–{(s['end'] or len(doc))}")
        else:
            print(f"  {s['subject']} (2011-2022) — pages {s['start']+1}–{s['end']}")

    all_records = []

    for s in sections:
        if s["type"] == "sequential":
            print(f"\nProcessing {s['year']} ...")
            recs = process_sequential(doc, s)
        else:
            print(f"\nProcessing {s['subject']} ...")
            recs = process_subjectwise(doc, s)

        ok      = sum(1 for r in recs if r["answer"])
        no_ans  = len(recs) - ok
        print(f"  Questions parsed     : {len(recs)}")
        print(f"  With answers         : {ok}")
        if no_ans:
            print(f"  Missing answers      : {no_ans}")

        all_records.extend(recs)

    doc.close()

    # ── Save ────────────────────────────────────────────────────────────────
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    with_ans  = sum(1 for r in all_records if r["answer"])
    with_exp  = sum(1 for r in all_records if r["explanation"])
    year_dist = {}
    for r in all_records:
        yr = r["year"]
        year_dist[yr] = year_dist.get(yr, 0) + 1

    print(f"\n{'═'*52}")
    print(f"  PARSE COMPLETE")
    print(f"  Total questions      : {len(all_records)}")
    print(f"  With answers         : {with_ans}")
    print(f"  With explanations    : {with_exp}")
    print(f"\n  Year distribution:")
    for yr, cnt in sorted(year_dist.items(), key=lambda x: (x[0] is None, x[0])):
        print(f"    {yr or 'unknown':>7} : {cnt}")
    print(f"\n  Saved to: {OUTPUT}")
    print(f"{'═'*52}")
    print("\nNext step → run ingest_pyq.py to load PYQs into ChromaDB.")


if __name__ == "__main__":
    main()
