# agent_generate.py
# CSE-GO Pipeline — Agentic Question Generation
#
# Uses Claude's tool_use API to run a multi-step agent that:
#   1. Searches NCERT chunks for evidence (can retry with different queries)
#   2. Retrieves real UPSC PYQ style examples for framing reference
#   3. Drafts a question grounded strictly in retrieved evidence
#   4. Self-critiques — if the extract doesn't cleanly support the answer, tries again
#   5. Saves the question when confident
#
# For question SET generation, the agent plans coverage across sub-topics,
# tracks what it has already generated, and ensures variety in question type
# and difficulty.
#
# ── HOW TO RUN ────────────────────────────────────────────────────────────────
#
#   # Generate a set of questions on a subject:
#   python agent_generate.py --subject "Environment and Ecology" --count 5
#
#   # Generate questions from a list of topics:
#   python agent_generate.py --topics "ozone layer" "wetlands" "carbon cycle"
#
#   # Single topic:
#   python agent_generate.py --topics "fundamental rights"
#
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import glob
import datetime
import argparse
import chromadb
import anthropic
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# ── CONFIGURATION ─────────────────────────────────────────────────────────────

CHROMA_DIR    = "./chroma-db"
NCERT_COL     = "cse_knowledge_base"
PYQ_COL       = "pyq_questions"
CA_COL        = "current_affairs"
EMBED_MODEL   = "all-MiniLM-L6-v2"
OUTPUT_DIR    = "./questions"
CLAUDE_MODEL  = "claude-sonnet-4-6"          # agentic mode default
SONNET_MODEL  = "claude-sonnet-4-6"          # CA + inference questions
HAIKU_MODEL   = "claude-haiku-4-5-20251001"  # static + PYQ-style questions

MAX_AGENT_TURNS = 14    # safety cap on tool calls per question
TOP_K_NCERT     = 5
TOP_K_PYQ       = 2
TOP_K_CA        = 5

# Batch mode: CA distance threshold — below this = CA hit → use Sonnet
CA_HIT_THRESHOLD = 0.55
# Batch mode: chunks retrieved per topic
BATCH_NCERT_K = 8
BATCH_CA_K    = 5


# ── TOOL DEFINITIONS ──────────────────────────────────────────────────────────
# These are the tools Claude can call during its reasoning loop.

TOOLS = [
    {
        "name": "search_ncert",
        "description": (
            "Search the NCERT knowledge base for text chunks relevant to a query. "
            "Returns the top matching chunks with their source file and page number. "
            "Use this to find factual evidence for a question. "
            "You can call this multiple times with different queries to find better evidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — a phrase or sentence describing the concept you want evidence for."
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of chunks to retrieve (default 3, max 5).",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_pyq",
        "description": (
            "Search past UPSC Prelims questions (2011-2025) for questions on a similar topic. "
            "Returns real UPSC questions as style and difficulty reference. "
            "Use this to understand how UPSC frames questions on this topic — "
            "adopt the same style (statement-based, match pairs, etc.) in your question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic or concept to search for in past UPSC questions."
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of past questions to retrieve (default 2, max 4).",
                    "default": 2
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_ca",
        "description": (
            "Search the current affairs knowledge base for recent events, government schemes, "
            "organizations, treaties, species, and policies relevant to a topic. "
            "Each result includes a reworded synthesis and structured UPSC-testable facts with "
            "concept links — use these to write questions that connect current events to static "
            "UPSC concepts. Call this when the topic has a current affairs dimension. "
            "You can combine CA evidence with NCERT evidence in a single question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic or event to search for in current affairs."
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of CA chunks to retrieve (default 3, max 5).",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "save_question",
        "description": (
            "Save a finalized question to the output set. "
            "Only call this when you are confident that: "
            "(1) the correct answer is explicitly stated in the cited_extract, "
            "(2) the question does not reference 'the passage', "
            "(3) all four options are present and plausible, "
            "(4) the cited_extract is a verbatim copy from the chunk. "
            "If you are not confident, search for better evidence first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question":       {"type": "string", "description": "The question text."},
                "options":        {
                    "type": "object",
                    "description": "Four options keyed A, B, C, D.",
                    "properties": {
                        "A": {"type": "string"},
                        "B": {"type": "string"},
                        "C": {"type": "string"},
                        "D": {"type": "string"}
                    },
                    "required": ["A", "B", "C", "D"]
                },
                "correct_answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "explanation":    {"type": "string", "description": "2-3 sentence explanation of why the answer is correct. Write as standalone factual prose — do NOT reference 'the chunk', 'the passage', 'the extract', 'CA chunk', 'NCERT text', or any source. Explain the fact directly as knowledge."},
                "cited_extracts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of verbatim sentences from source chunks — one per statement or claim in the question. For a 3-statement question, provide 3 extracts, one proving or disproving each statement."
                },
                "source_file":    {"type": "string", "description": "The source filename or CA topic label."},
                "source_page":    {"type": "integer", "description": "Page number in the source file (use 0 for CA sources)."},
                "source_type":    {
                    "type": "string",
                    "enum": ["ncert", "ca", "both"],
                    "description": "Whether evidence came from NCERT, current affairs, or both."
                },
                "question_type":  {
                    "type": "string",
                    "enum": ["statement_based", "match_pairs", "single_fact", "assertion_reason", "how_many"],
                    "description": "Category of question format."
                },
                "difficulty":     {
                    "type": "string",
                    "enum": ["easy", "medium", "hard"],
                    "description": "Estimated difficulty for a UPSC aspirant."
                }
            },
            "required": ["question", "options", "correct_answer", "explanation",
                         "cited_extracts", "source_file", "source_page", "source_type",
                         "question_type", "difficulty"]
        }
    }
]


# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert UPSC Civil Services Examination (Prelims) question setter with 15 years of experience.

Your task is to generate ONE high-quality UPSC Prelims MCQ question on the given topic.

YOU HAVE THREE KNOWLEDGE SOURCES — use them strategically:
  search_ncert  → foundational concepts, definitions, classifications, processes (NCERT Class 6-12)
  search_ca     → current affairs: recent schemes, organizations, treaties, species, policies
  search_pyq    → real UPSC questions (2011-2025) — use ONLY for style/framing reference

THE BEST UPSC QUESTIONS connect a current event to a static concept:
  e.g. "A new Ramsar site was declared in 2025" → question on Ramsar Convention criteria (NCERT/static)
  e.g. "Government launched PM-Surya Ghar scheme" → question on solar energy / ministry / beneficiary
If the topic has a current affairs angle, call search_ca first to find the event/scheme, then
call search_ncert to find the underlying concept that makes it UPSC-worthy.

YOUR PROCESS:
1. If topic sounds current-affairs-related → search_ca first, then search_ncert for the concept layer.
   If topic sounds purely conceptual → search_ncert first (2-3 queries if needed).
2. Use search_pyq to see how UPSC frames questions on this topic — adopt that style.
3. Draft a question mentally. Check: can the correct answer be directly proven from the evidence?
4. If yes → call save_question. If no → search again with a different query.

STRICT RULES for the question you save:
- cited_extracts is an ARRAY — one verbatim sentence per statement/claim in the question.
  For a 3-statement question, provide 3 extracts. For a single-fact question, provide 1.
  Each extract must be VERBATIM text from the retrieved chunk — no paraphrasing.
  (For CA sources the chunk is already a synthesis — quote it as-is.)
- Every statement marked correct/incorrect must be provable from its cited extract.
- Do NOT write "According to the passage" or reference the source.
- Distractors must be plausible but clearly wrong based on the evidence.
- Match question style (statement-based, match pairs, etc.) to what UPSC uses for this topic.
- Do not duplicate a question already in the PYQ search results.
- Set source_type = "ncert" / "ca" / "both" to reflect where evidence came from.

ASSERTION-REASON RESTRICTION:
Only use assertion_reason format for Science & Technology, Biology, Chemistry, or Physics topics
where causal relationships are scientifically definitive.
For ALL other subjects (History, Polity, Geography, Economy, Environment, Current Affairs,
Governance, IR) — use statement_based or single_fact. Never use AR for these subjects.

DIFFICULTY GUIDE:
- easy: single direct fact, no inference needed
- medium: requires comparing statements or matching, 1-2 step reasoning
- hard: subtle distinction, common misconception as distractor, requires careful reading\
"""


# ── TOOL EXECUTORS ────────────────────────────────────────────────────────────

def execute_search_ncert(ncert_col, query, n_results=5, subject_filter=None):
    n = min(int(n_results), 8)
    # Try subject-scoped retrieval first; fall back to unfiltered if too few results
    where = {"subject": {"$eq": subject_filter}} if subject_filter else None
    try:
        results = ncert_col.query(query_texts=[query], n_results=n, where=where)
        if where and len(results["documents"][0]) < 2:
            # Not enough subject-scoped chunks — widen to full collection
            results = ncert_col.query(query_texts=[query], n_results=n)
    except Exception:
        results = ncert_col.query(query_texts=[query], n_results=n)
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text":     results["documents"][0][i],
            "source":   results["metadatas"][0][i]["source"],
            "page":     results["metadatas"][0][i]["page"],
            "subject":  results["metadatas"][0][i].get("subject", ""),
            "distance": round(results["distances"][0][i], 4)
        })
    return chunks


def execute_search_pyq(pyq_col, query, n_results=2):
    n = min(int(n_results), 4)
    results = pyq_col.query(query_texts=[query], n_results=n)
    questions = []
    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i]
        opts = json.loads(meta.get("options_json", "{}"))
        questions.append({
            "question":    results["documents"][0][i],
            "options":     opts,
            "answer":      meta.get("answer"),
            "year":        meta.get("year"),
            "subject":     meta.get("subject"),
            "explanation": meta.get("explanation", "")[:300]
        })
    return questions


def execute_search_ca(ca_col, query, n_results=3):
    n = min(int(n_results), 5)
    results = ca_col.query(query_texts=[query], n_results=n)
    chunks = []
    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i]
        facts_raw = meta.get("upsc_facts", "[]")
        try:
            facts = json.loads(facts_raw)
        except (json.JSONDecodeError, TypeError):
            facts = []
        chunks.append({
            "text":     results["documents"][0][i],
            "topic":    meta.get("topic", ""),
            "category": meta.get("category", ""),
            "date":     meta.get("date", ""),
            "upsc_facts": facts,
            "distance": round(results["distances"][0][i], 4)
        })
    return chunks


def dispatch_tool(tool_name, tool_input, ncert_col, pyq_col, ca_col, saved_questions):
    """Execute the tool and return a JSON-serialisable result dict."""
    if tool_name == "search_ncert":
        results = execute_search_ncert(
            ncert_col,
            tool_input["query"],
            tool_input.get("n_results", TOP_K_NCERT)
        )
        return {"chunks": results, "count": len(results)}

    elif tool_name == "search_ca":
        if ca_col is None:
            return {"error": "Current affairs collection not available. Use search_ncert instead."}
        results = execute_search_ca(
            ca_col,
            tool_input["query"],
            tool_input.get("n_results", TOP_K_CA)
        )
        return {"chunks": results, "count": len(results)}

    elif tool_name == "search_pyq":
        results = execute_search_pyq(
            pyq_col,
            tool_input["query"],
            tool_input.get("n_results", TOP_K_PYQ)
        )
        return {"past_questions": results, "count": len(results)}

    elif tool_name == "save_question":
        saved_questions.append(tool_input)
        return {"status": "saved", "total_saved": len(saved_questions)}

    return {"error": f"Unknown tool: {tool_name}"}


# ── AGENT LOOP ────────────────────────────────────────────────────────────────

def run_agent(client, topic, ncert_col, pyq_col, ca_col=None, context_note=""):
    """
    Runs the agent loop for a single topic.
    Returns the saved question dict, or None if the agent couldn't produce one.
    """
    saved_questions = []

    user_content = f"Generate one UPSC Prelims MCQ question on this topic: **{topic}**"
    if context_note:
        user_content += f"\n\nAdditional context: {context_note}"

    messages = [{"role": "user", "content": user_content}]

    print(f"    Agent thinking", end="", flush=True)

    for turn in range(MAX_AGENT_TURNS):
        response = client.messages.create(
            model      = CLAUDE_MODEL,
            max_tokens = 2048,
            system     = [{"type": "text", "text": SYSTEM_PROMPT,
                           "cache_control": {"type": "ephemeral"}}],
            tools      = TOOLS,
            messages   = messages
        )

        # Collect all tool calls and text from this response
        tool_calls = [b for b in response.content if b.type == "tool_use"]
        has_text   = any(b.type == "text" for b in response.content)

        print(".", end="", flush=True)

        # Append assistant message
        messages.append({"role": "assistant", "content": response.content})

        # If agent is done (no tool calls), stop
        if response.stop_reason == "end_turn" and not tool_calls:
            break

        if not tool_calls:
            break

        # Execute all tool calls and build the tool_result message
        tool_results = []
        for tc in tool_calls:
            result = dispatch_tool(tc.name, tc.input, ncert_col, pyq_col, ca_col, saved_questions)
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": tc.id,
                "content":     json.dumps(result)
            })

        messages.append({"role": "user", "content": tool_results})

        # If save_question was called, we're done
        if any(tc.name == "save_question" for tc in tool_calls):
            break

    print()  # newline after dots

    return saved_questions[0] if saved_questions else None


# ── BATCH MODE ────────────────────────────────────────────────────────────────
# Batch generation: retrieve chunks locally (free), then one LLM call per tier.
# Sonnet tier: topics with a CA hit or requiring inference/synthesis.
# Haiku tier:  static NCERT topics (single-fact, statement-based, PYQ-style).

BATCH_SYSTEM_STATIC = """\
You are an expert UPSC Civil Services Examination (Prelims) question setter.

You will be given several topics, each with retrieved NCERT text chunks as evidence.
Generate ONE high-quality MCQ for EACH topic. Preferred format: statement_based.
Only use assertion_reason format if the topic is explicitly Science & Technology, Biology, Chemistry, or Physics — where causal relationships are definitive and unambiguous. For all other subjects (History, Polity, Geography, Economy, Environment, Current Affairs) use statement_based or single_fact.

── QUESTION DESIGN PRINCIPLES ───────────────────────────────────────────────────
Test UNDERSTANDING and RECALL of concepts, processes, and relationships — not rote memorisation.

AVOID:
  • Options that differ only by exact numbers, percentages, or specific dates
    (e.g. do NOT ask "which state has X tigers" if options are 450/512/623/731)
  • Trick questions built on minor numerical distinctions the candidate cannot be expected to know
  • Obscure trivia absent from mainstream UPSC preparation material

PREFER:
  • Statements about HOW or WHY something works (constitutional mechanisms, ecological processes)
  • Comparisons that test conceptual distinction (e.g. core zone vs buffer zone, not exact area figures)
  • Statements where incorrectness is conceptually meaningful (wrong principle, not wrong number)

── CARDINAL RULE: EVERY STATEMENT NEEDS AN EXTRACT ─────────────────────────────
For EACH statement you write in the question body:
  • If the statement is CORRECT  → provide a verbatim extract from the chunk that
    explicitly confirms it.
  • If the statement is INCORRECT → provide a verbatim extract from the chunk that
    explicitly contradicts it (the chunk says X, your statement falsely says Y).

Do NOT include a statement if you cannot find a verbatim extract to confirm or contradict it.
Do NOT test facts that only exist in your training knowledge — only what the chunks say.
Do NOT reference "the passage", "the chunk", "the extract", "CA chunk", "NCERT text", or any source in the question OR explanation. Write all explanations as direct factual statements.
Do NOT frame questions as "Based on NCERT evidence", "According to NCERT", or "As per NCERT". NCERTs are textbooks, not primary historical sources — questions must stand on factual merit alone.

── cited_extracts ARRAY ─────────────────────────────────────────────────────────
One entry per statement, in the same order as the statements appear in the question.
Each entry = verbatim text copied from the chunk. No paraphrasing.

Return ONLY a valid JSON array (one object per topic), no markdown fences:
[
  {
    "topic_query": "...",
    "question": "Consider the following statements:\\n1. [correct statement]\\n2. [incorrect statement]\\n3. [correct statement]\\nWhich of the above is/are correct?",
    "options": {"A":"1 only","B":"1 and 3 only","C":"2 and 3 only","D":"1, 2 and 3"},
    "correct_answer": "B",
    "explanation": "Statement 1 is correct: [state the fact directly]. Statement 2 is incorrect: [state why it is wrong as a fact]. Statement 3 is correct: [state the fact directly].",
    "cited_extracts": [
      "Verbatim chunk text CONFIRMING statement 1.",
      "Verbatim chunk text CONTRADICTING statement 2.",
      "Verbatim chunk text CONFIRMING statement 3."
    ],
    "source_file": "filename",
    "source_page": 0,
    "source_type": "ncert",
    "question_type": "statement_based",
    "difficulty": "medium"
  }
]\
"""

BATCH_SYSTEM_CA = """\
You are an expert UPSC Civil Services Examination (Prelims) question setter.

You will be given several topics, each with:
  - Current affairs chunks (recent events, schemes, policies) with UPSC facts + concept links
  - NCERT chunks providing the underlying static concept

Generate ONE MCQ per topic that CONNECTS the current event to the underlying UPSC concept.
Preferred format: statement_based.
Only use assertion_reason if the topic is Science & Technology, Biology, Chemistry, or Physics — where the causal relationship between A and R is scientifically definitive. For all other subjects use statement_based.
Difficulty: medium to hard.

── QUESTION DESIGN PRINCIPLES ───────────────────────────────────────────────────
Test UNDERSTANDING of mechanisms, principles, and significance — not rote recall of specifics.

AVOID:
  • Options that differ only by exact numbers, percentages, dates, or rankings
    (e.g. do NOT ask "what percentage of tariff lines are covered" if options are 96/97/98/99%)
  • Questions where a candidate must recall a specific figure to distinguish options
  • Obscure statistics unlikely to appear in mainstream UPSC preparation

PREFER:
  • Questions testing WHY a policy/scheme/event matters constitutionally or conceptually
  • Statements testing correct understanding of a process (how does X work, not what is the number)
  • Assertion-reason pairs where the relationship between event and principle is the key insight
  • Comparing two related concepts the current event illustrates (e.g. Ramsar criteria vs sanctuary vs reserve)

── CARDINAL RULE: EVERY CLAIM NEEDS AN EXTRACT ──────────────────────────────────
For EACH statement or claim in the question:
  • CORRECT claim  → verbatim extract from a chunk that explicitly confirms it.
  • INCORRECT claim → verbatim extract from a chunk that explicitly contradicts it.

For assertion_reason format specifically:
  • Extract 1 must confirm or contradict the Assertion (A) from a CA chunk.
  • Extract 2 must confirm or contradict the Reason (R) from an NCERT or CA chunk.
  Only write A or R if you have a verbatim extract to ground it.

For statement_based format:
  One extract per statement, same order as statements appear. Confirm OR contradict.

Do NOT test facts absent from the provided chunks.
Do NOT reference "the passage", "the chunk", "the extract", "CA chunk", "NCERT text", or any source in the question OR explanation. Write all explanations as direct factual statements.
Do NOT frame questions as "Based on NCERT evidence", "According to NCERT", or "As per NCERT". NCERTs are textbooks, not primary sources — questions must stand on factual merit alone.
Use concept_link notes from the CA facts to guide what underlying UPSC concept to test.

── ASSERTION-REASON RESTRICTION ─────────────────────────────────────────────────
Only use assertion_reason for Science & Technology topics where causal relationships are
scientifically definitive. For all other subjects — including Current Affairs, Governance,
IR, Economy, History, Polity — use statement_based. Never use AR for policy or event topics.

Return ONLY a valid JSON array (one object per topic), no markdown fences:
[
  {
    "topic_query": "...",
    "question": "...",
    "options": {"A":"...","B":"...","C":"...","D":"..."},
    "correct_answer": "A",
    "explanation": "For each claim, state which extract confirms or contradicts it. Reference both CA event and static concept.",
    "cited_extracts": [
      "Verbatim CA chunk text confirming/contradicting claim 1.",
      "Verbatim NCERT chunk text confirming/contradicting claim 2.",
      "Additional verbatim extract if needed."
    ],
    "source_file": "CA topic label or NCERT filename",
    "source_page": 0,
    "source_type": "both",
    "question_type": "assertion_reason",
    "difficulty": "hard"
  }
]\
"""


def batch_retrieve(topics, ncert_col, ca_col):
    """
    For each topic, retrieve NCERT and CA chunks locally (no LLM call).
    Returns list of {topic, ncert_chunks, ca_chunks, tier} dicts.
    Tier = 'sonnet' if CA hit is strong, else 'haiku'.
    """
    # Map UPSC subject names to metadata subject values used in ingest.py
    SUBJECT_MAP = {
        "history": "History", "art & culture": "History",
        "geography": "Geography",
        "polity": "Polity", "governance": "Polity",
        "economics": "Economics", "economy": "Economics",
        "science": "Science & Technology", "science & technology": "Science & Technology",
        "environment": "Geography",  # env content mostly in Geography NCERTs
        "current affairs": None,     # no subject filter for CA topics
    }

    def _subject_for_topic(topic_str):
        t = topic_str.lower()
        for key, val in SUBJECT_MAP.items():
            if key in t:
                return val
        return None

    results = []
    for topic in topics:
        subject_hint = _subject_for_topic(topic)
        where = {"subject": {"$eq": subject_hint}} if subject_hint else None
        try:
            ncert_res = ncert_col.query(query_texts=[topic], n_results=BATCH_NCERT_K, where=where)
            if where and len(ncert_res["documents"][0]) < 2:
                ncert_res = ncert_col.query(query_texts=[topic], n_results=BATCH_NCERT_K)
        except Exception:
            ncert_res = ncert_col.query(query_texts=[topic], n_results=BATCH_NCERT_K)
        ncert_chunks = []
        for i in range(len(ncert_res["documents"][0])):
            ncert_chunks.append({
                "text":     ncert_res["documents"][0][i],
                "source":   ncert_res["metadatas"][0][i].get("source", ""),
                "page":     ncert_res["metadatas"][0][i].get("page", 0),
                "distance": round(ncert_res["distances"][0][i], 4),
            })

        ca_chunks = []
        tier = "haiku"
        if ca_col:
            ca_res = ca_col.query(query_texts=[topic], n_results=BATCH_CA_K)
            for i in range(len(ca_res["documents"][0])):
                dist = ca_res["distances"][0][i]
                meta = ca_res["metadatas"][0][i]
                try:
                    facts = json.loads(meta.get("upsc_facts", "[]"))
                except (json.JSONDecodeError, TypeError):
                    facts = []
                ca_chunks.append({
                    "text":       ca_res["documents"][0][i],
                    "topic":      meta.get("topic", ""),
                    "category":   meta.get("category", ""),
                    "upsc_facts": facts,
                    "distance":   round(dist, 4),
                })
            if ca_chunks and ca_chunks[0]["distance"] < CA_HIT_THRESHOLD:
                tier = "sonnet"

        results.append({
            "topic":        topic,
            # Haiku tier = NCERT-only; don't pass CA chunks to avoid retrieval bleed
            "ncert_chunks": ncert_chunks,
            "ca_chunks":    ca_chunks if tier == "sonnet" else [],
            "tier":         tier,
        })
    return results


def _format_topic_block(item):
    """Format one topic's retrieved chunks into a text block for the batch prompt."""
    lines = [f"TOPIC: {item['topic']}"]

    if item["ca_chunks"]:
        lines.append("\nCURRENT AFFAIRS EVIDENCE:")
        for c in item["ca_chunks"]:
            lines.append(f"  [CA | {c['topic']} | {c['category']}]")
            lines.append(f"  {c['text'][:800]}")
            if c["upsc_facts"]:
                lines.append("  UPSC Facts:")
                for f in c["upsc_facts"][:3]:
                    if isinstance(f, dict):
                        lines.append(f"    • {f.get('fact','')} → {f.get('concept_link','')}")
                    else:
                        lines.append(f"    • {f}")

    lines.append("\nNCERT EVIDENCE:")
    for c in item["ncert_chunks"][:4]:
        lines.append(f"  [NCERT | {c['source']} | p.{c['page']}]")
        lines.append(f"  {c['text'][:700]}")

    return "\n".join(lines)


def batch_generate(client, tier_items, tier):
    """
    Single LLM call to generate one question per topic in tier_items.
    tier = 'sonnet' | 'haiku'
    Returns list of question dicts.
    """
    model  = SONNET_MODEL if tier == "sonnet" else HAIKU_MODEL
    system = BATCH_SYSTEM_CA if tier == "sonnet" else BATCH_SYSTEM_STATIC

    topic_blocks = "\n\n" + ("═" * 60) + "\n\n"
    topic_blocks = topic_blocks.join(_format_topic_block(item) for item in tier_items)

    user_msg = (
        f"Generate exactly {len(tier_items)} MCQ question(s), one per topic.\n\n"
        f"{topic_blocks}"
    )

    # Split large batches to avoid context/token limits
    BATCH_CHUNK = 5
    if len(tier_items) > BATCH_CHUNK:
        all_questions = []
        for i in range(0, len(tier_items), BATCH_CHUNK):
            all_questions.extend(
                batch_generate(client, tier_items[i:i + BATCH_CHUNK], tier)
            )
        return all_questions

    response = client.messages.create(
        model      = model,
        max_tokens = 600 * len(tier_items) + 512,
        system     = [{"type": "text", "text": system,
                       "cache_control": {"type": "ephemeral"}}],
        messages   = [{"role": "user", "content": user_msg}]
    )

    raw = response.content[0].text.strip()
    import re

    def _extract_json_array(text):
        """Try progressively looser JSON extraction."""
        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Strip markdown fences
        cleaned = re.sub(r'^```json\s*|^```\s*|```\s*$', '', text, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Extract first [...] block
        m = re.search(r'(\[.*\])', cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        return None

    questions = _extract_json_array(raw)
    if questions is None:
        print(f"\n  WARNING: Could not parse {tier} batch response. Raw output:")
        print(raw[:500])
        return []

    # Tag each question with tier and status
    for q, item in zip(questions, tier_items):
        q["topic_query"] = item["topic"]
        q["model_tier"]  = tier
        q["status"]      = "pending_check"

    # Self-verify: strip questions where extracts don't support their statements
    questions = _self_verify(client, questions, model)

    return questions


SELF_VERIFY_PROMPT = """\
You are a strict quality checker for UPSC MCQ questions.

For each question below, check every cited_extract against its corresponding statement:
  - Does the extract contain language that EXPLICITLY confirms or contradicts the statement?
  - A heading, label, or vague paraphrase does NOT count.
  - An extract about a different topic does NOT count.

Return a JSON array with one object per question:
[
  { "topic_query": "...", "valid": true/false, "reason": "one line if invalid, empty string if valid" }
]

Only mark valid=false if there is a clear mismatch. Be strict but fair.
Return ONLY the JSON array, no markdown.
"""

def _self_verify(client, questions, model):
    """Flag questions whose extracts don't support their statements. Returns all questions."""
    if not questions:
        return questions

    payload = [
        {
            "topic_query": q.get("topic_query", ""),
            "question": q.get("question", "")[:400],
            "cited_extracts": q.get("cited_extracts", []),
            "correct_answer": q.get("correct_answer", ""),
        }
        for q in questions
    ]

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            system=SELF_VERIFY_PROMPT,
            messages=[{"role": "user", "content": json.dumps(payload)}],
        )
        import re
        raw = resp.content[0].text.strip()
        raw = re.sub(r'^```json\s*|^```\s*|```\s*$', '', raw, flags=re.MULTILINE).strip()
        verdicts = json.loads(raw)
    except Exception as e:
        print(f"\n  [self-verify] skipped ({e})")
        return questions

    verdict_map = {v["topic_query"]: v for v in verdicts}
    invalid = [v for v in verdicts if not v.get("valid", True)]

    if invalid:
        print(f"\n  [self-verify] flagged {len(invalid)} question(s) with mismatched extracts:")
        for v in invalid:
            print(f"    ✗ {v['topic_query']}: {v.get('reason','')}")

    for q in questions:
        v = verdict_map.get(q.get("topic_query", ""))
        if v and not v.get("valid", True):
            q["status"] = "flag"
            q["flag_reason"] = f"Self-verify: {v.get('reason', 'extract does not support statements')}"

    return questions


# ── QUESTION SET PLANNER ──────────────────────────────────────────────────────

PLANNER_SYSTEM = """\
You are a UPSC syllabus expert. Given a subject and a count N, produce a JSON list of N topic-queries
that together give good coverage of that subject for UPSC Prelims.

Rules:
- Each query should be specific enough to retrieve focused NCERT chunks.
- Vary the sub-topics — don't repeat the same concept.
- Vary the expected question types: some should invite statement-based questions,
  some match-pairs, some single-fact.
- Return ONLY a valid JSON array of strings, no explanation, no markdown fences.

Example output for subject="Indian Polity", count=3:
["fundamental rights article 19 freedoms", "directive principles of state policy", "parliamentary privileges speaker"]
\
"""

def plan_question_set(client, subject, count):
    """Ask Claude to plan N diverse topic queries for a subject."""
    response = client.messages.create(
        model      = CLAUDE_MODEL,
        max_tokens = 512,
        system     = PLANNER_SYSTEM,
        messages   = [{
            "role": "user",
            "content": f"Subject: {subject}\nCount: {count}\n\nReturn a JSON array of {count} specific topic queries."
        }]
    )
    raw = response.content[0].text.strip()
    try:
        topics = json.loads(raw)
        return topics[:count]
    except json.JSONDecodeError:
        import re
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            return json.loads(m.group())[:count]
        return [f"{subject} topic {i+1}" for i in range(count)]


# ── OUTPUT ────────────────────────────────────────────────────────────────────

def _existing_topics(output_dir):
    """Return set of (normalised) topic_query strings already saved across all batch files."""
    seen = set()
    for fpath in glob.glob(os.path.join(output_dir, "*.json")):
        try:
            with open(fpath) as f:
                data = json.load(f)
            if isinstance(data, list):
                for q in data:
                    tq = q.get("topic_query") or ""
                    seen.add(tq.lower().strip())
        except Exception:
            pass
    return seen


def deduplicate(records, output_dir):
    """Remove records whose topic_query closely matches an already-saved question."""
    existing = _existing_topics(output_dir)
    unique, dupes = [], []
    for q in records:
        tq = (q.get("topic_query") or "").lower().strip()
        if tq in existing:
            dupes.append(tq)
        else:
            unique.append(q)
            existing.add(tq)   # prevent duplicates within this batch too
    return unique, dupes


def save_batch(records, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    unique, dupes = deduplicate(records, output_dir)
    if dupes:
        print(f"  Deduplication: removed {len(dupes)} repeat topic(s): {dupes}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = os.path.join(output_dir, f"agent_batch_{timestamp}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)
    return filepath, unique


def print_question(q, index):
    if not q:
        return
    print(f"\n{'─'*60}")
    print(f"  Q{index+1}. [{q.get('question_type','?')} | {q.get('difficulty','?')}]")
    print(f"  {q.get('question','')[:120]}{'…' if len(q.get('question',''))>120 else ''}")
    opts = q.get("options", {})
    for k in ["A","B","C","D"]:
        marker = "✓" if k == q.get("correct_answer") else " "
        print(f"    {marker} {k}) {opts.get(k,'')}")
    print(f"\n  Explanation: {q.get('explanation','')[:150]}…")
    print(f"  Source: {q.get('source_file','')}  p.{q.get('source_page','')}")
    extracts = q.get('cited_extracts') or ([q.get('cited_extract','')] if q.get('cited_extract') else [])
    for ex in extracts[:3]:
        print(f"  Extract: \"{str(ex)[:100]}…\"")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Agentic UPSC MCQ generator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--topics", nargs="+", help="One or more topic queries")
    group.add_argument("--subject", help="Subject name — agent plans coverage automatically")
    parser.add_argument("--count", type=int, default=5,
                        help="Number of questions to generate (used with --subject, default 5)")
    parser.add_argument("--batch", action="store_true",
                        help="Batch mode: retrieve locally then one LLM call per tier (cheaper). "
                             "Sonnet for CA/inference topics, Haiku for static NCERT topics.")
    args = parser.parse_args()

    # ── Check API key ─────────────────────────────────────────────────────────
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nERROR: ANTHROPIC_API_KEY not set.")
        print("Run:  export ANTHROPIC_API_KEY='sk-ant-...'")
        sys.exit(1)

    # ── Load ChromaDB ─────────────────────────────────────────────────────────
    if not os.path.exists(CHROMA_DIR):
        print(f"\nERROR: ChromaDB not found at '{CHROMA_DIR}'.")
        print("Run ingest.py and ingest_pyq.py first.")
        sys.exit(1)

    client_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn      = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    ncert_col = client_chroma.get_or_create_collection(NCERT_COL, embedding_function=embed_fn)
    pyq_col   = client_chroma.get_or_create_collection(PYQ_COL,   embedding_function=embed_fn)

    # CA collection is optional — only available after running ingest_ca.py
    try:
        ca_col = client_chroma.get_collection(CA_COL, embedding_function=embed_fn)
        ca_count = ca_col.count()
    except Exception:
        ca_col   = None
        ca_count = 0

    print(f"NCERT chunks   : {ncert_col.count()}")
    print(f"PYQ questions  : {pyq_col.count()}")
    print(f"CA chunks      : {ca_count}" + ("" if ca_count else "  (run ingest_ca.py to add)"))

    anthropic_client = anthropic.Anthropic()

    # ── Plan topics ───────────────────────────────────────────────────────────
    if args.subject:
        print(f"\nPlanning {args.count} questions covering '{args.subject}' ...")
        topics = plan_question_set(anthropic_client, args.subject, args.count)
        print(f"Topics planned:")
        for t in topics:
            print(f"  • {t}")
    else:
        topics = args.topics

    # ── Generate questions ────────────────────────────────────────────────────
    records = []
    ok = 0

    if args.batch:
        print(f"\nBatch mode — retrieving chunks for {len(topics)} topic(s) ...")
        retrieved = batch_retrieve(topics, ncert_col, ca_col)

        sonnet_items = [r for r in retrieved if r["tier"] == "sonnet"]
        haiku_items  = [r for r in retrieved if r["tier"] == "haiku"]

        print(f"  Sonnet tier (CA/inference): {len(sonnet_items)} topics")
        print(f"  Haiku tier  (static NCERT): {len(haiku_items)} topics")

        if haiku_items:
            print(f"\nGenerating {len(haiku_items)} static question(s) with Haiku ...")
            haiku_qs = batch_generate(anthropic_client, haiku_items, "haiku")
            for i, q in enumerate(haiku_qs):
                records.append(q)
                print_question(q, len(records) - 1)
                ok += 1

        if sonnet_items:
            print(f"\nGenerating {len(sonnet_items)} CA/inference question(s) with Sonnet ...")
            sonnet_qs = batch_generate(anthropic_client, sonnet_items, "sonnet")
            for i, q in enumerate(sonnet_qs):
                records.append(q)
                print_question(q, len(records) - 1)
                ok += 1

    else:
        print(f"\nGenerating {len(topics)} question(s) (agentic mode) ...\n")

        for i, topic in enumerate(topics):
            print(f"[{i+1}/{len(topics)}] Topic: '{topic}'")
            q = run_agent(anthropic_client, topic, ncert_col, pyq_col, ca_col)

            if q:
                q["topic_query"] = topic
                q["status"]      = "pending_check"
                records.append(q)
                print_question(q, i)
                ok += 1
            else:
                print(f"    ✗ Agent could not produce a question for this topic.")
                records.append({
                    "topic_query": topic,
                    "status":      "agent_failed",
                    "question":    None
                })

    # ── Save ──────────────────────────────────────────────────────────────────
    output_file, saved = save_batch(records, OUTPUT_DIR)

    print(f"\n{'═'*52}")
    print(f"  GENERATION COMPLETE")
    print(f"  Questions generated  : {ok} / {len(topics)}")
    print(f"  Saved (after dedup)  : {len(saved)}")
    print(f"  Saved to             : {output_file}")
    print(f"{'═'*52}")
    print("\nNext step → run check.py to verify the questions.")


if __name__ == "__main__":
    main()
