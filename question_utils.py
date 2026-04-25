"""
question_utils.py — Shared deduplication helpers used by daily_pipeline.py and api.py.
"""

import re as _re

STOPWORDS = {"the","of","in","and","to","a","an","for","on","with",
             "india","indian","its","their","is","are","was","were","has"}

CONCEPT_TRIGGERS = {
    "index","report","convention","treaty","act","scheme","programme","program",
    "mission","initiative","summit","policy","framework","commission","committee",
    "amendment","protocol","declaration","court","tribunal","fund","bank","agency",
    "authority","board","council","institute","award","prize","medal",
}

SKIP_ACRONYMS = {"MCQ","UPSC","CSE","IAS","GS","OBC","SC","ST","UN","EU",
                 "GDP","GNP","GNI","WHO","WTO","IMF","WEF","RBI","SEBI"}


def topic_fingerprint(raw: str) -> str:
    """First 2 meaningful words of a topic label — primary dedup key."""
    words = [w.lower() for w in (raw or "").replace("-", " ").split()
             if w.lower() not in STOPWORDS and len(w) > 2]
    return " ".join(words[:2])


def question_fingerprint(qtext: str) -> str:
    """First 50 chars of question lowercased — catches same question with different topic labels."""
    return (qtext or "").lower().strip()[:50]


def extract_entities(text: str) -> set:
    """
    Extract named entities that must not repeat across a daily set:
      - Constitution article references: Article 19, Article 21A
      - Named acronyms: MGNREGA, PMGSY, CAMPA ...
      - Named concept phrases: "Ramsar Convention", "Human Development Index", "PM RAHAT Scheme"
    """
    entities = set()
    t = text or ""

    for m in _re.finditer(r'\bArticle\s+\d+[A-Z]?\b', t, _re.IGNORECASE):
        entities.add(m.group().lower())

    for m in _re.finditer(r'\b[A-Z]{2,}\b', t):
        if m.group() not in SKIP_ACRONYMS:
            entities.add(m.group())

    words = t.split()
    for i, w in enumerate(words):
        clean = w.strip('.,;:()[]"\'').lower()
        if clean in CONCEPT_TRIGGERS and i > 0:
            start = max(0, i - 4)
            phrase_words = words[start:i + 1]
            phrase = " ".join(pw.strip('.,;:()[]"\'') for pw in phrase_words)
            if any(pw[0].isupper() for pw in phrase_words[:-1] if pw):
                entities.add(phrase.lower())

    return entities


def pick_diverse_set(candidates, limit=10, recent_topic_fps=None, strict_entity=True):
    """
    Select up to `limit` questions from candidates applying:
      1. Topic fingerprint dedup (within set)
      2. Question text fingerprint dedup (within set)
      3. Entity dedup — no two questions share a proper noun / scheme / article (strict pass)
      4. recent_topic_fps — skip topics already covered in last N daily sets

    candidates: list of dicts with keys: id, topic_key, question
    Returns: (selected_ids, seen_topic_fps, seen_q_fps, seen_entities)
    """
    recent_fps  = recent_topic_fps or set()
    seen_t_fps  = set()
    seen_q_fps  = set()
    seen_ents   = set()
    picked      = []

    for c in candidates:
        fp      = topic_fingerprint(c.get("topic_key") or "")
        qfp     = question_fingerprint(c.get("question") or "")
        ents    = extract_entities(c.get("question") or "")

        if fp and fp in recent_fps:
            continue
        if (fp and fp in seen_t_fps) or (qfp and qfp in seen_q_fps):
            continue
        if strict_entity and (ents & seen_ents):
            continue

        picked.append(c["id"])
        if fp:   seen_t_fps.add(fp)
        if qfp:  seen_q_fps.add(qfp)
        seen_ents.update(ents)

        if len(picked) == limit:
            break

    return picked, seen_t_fps, seen_q_fps, seen_ents
