# CSE-GO Pipeline — Roadmap

## ✅ Shipped

| Feature | Notes |
|---------|-------|
| NCERT ingestion (5,260 chunks, Class 6–12) | Chapter-by-chapter download + merge |
| PYQ ingestion (1,197 questions, 2011–2025) | Parsed from Physics Wallah PDF |
| Agentic question generation | Tool-use loop: search_ncert + search_pyq + save_question |
| Batch generation with tiered models | Sonnet (CA/inference) + Haiku (static NCERT), ~20× cheaper |
| Current affairs ingestion | Reword agent (Haiku) → structured UPSC facts + concept links |
| CA + NCERT cross-collection generation | Agent searches both, connects events to static concepts |
| Adversarial check (check.py) | Second Claude call as hostile examiner, grounding/ambiguity/distractors |
| Terminal quiz viewer (quiz.py) | PYQs + generated questions, interactive + review modes |
| cited_extracts array | One verbatim extract per statement (confirm or contradict) |

---

## 🔜 Next Up

### Question Quality
- [ ] **Assertion-reason logical validation** — micro Haiku call after generation to verify R causally explains A, not just correlates. Option: restrict AR to agentic mode only for now. ~1 day, ~$0.01/question.
- [ ] **Fix reword failures** — 34–47 chunks per PDF fall back to raw text. Likely Haiku timeouts on dense chunks. Add chunking pre-filter or retry logic.
- [ ] **Filter low-content CA chunks** — December 2025 PDF was a table of contents; chunks are index entries not substantive text. Add minimum word/sentence density filter before reword.

### Coverage
- [ ] **More CA months** — add remaining Vision IAS months (currently Dec 2025, Jan 2026, Feb 2026 only). Target: 12 months rolling.
- [ ] **PIB integration** — `ingest_ca.py --pib` is built but untested end-to-end. Test with 14-day pull.
- [ ] **Subject coverage audit** — run `agent_generate.py --subject` across all UPSC Prelims subjects, identify gaps in NCERT chunk retrieval.

### Pipeline
- [ ] **`--subject` batch mode** — currently `--subject` only works in agentic mode. Wire planner to batch mode.
- [ ] **Daily question set automation** — cron/scheduler to generate N questions per day and write to `questions/daily/`.
- [ ] **Question deduplication** — before saving, check new question against existing batches to avoid repeats on same topic.

---

## 🗂 Parked (discuss before building)

### Product
- [ ] **Lovable-based frontend** — user said defer. Needs: question display, answer reveal, score tracking, subject filter. API layer needed first.
- [ ] **API layer** — FastAPI or similar to serve questions and accept answers. Prerequisite for any frontend.
- [ ] **User progress tracking** — which questions attempted, weak subjects, accuracy over time.

### Quality
- [ ] **Assertion-reason format (option 2)** — logical validation micro-call. Parked until pass rate is stable on statement_based.
- [ ] **Assertion-reason format (option 3)** — only generate AR when source text contains explicit causal language. More elegant but more work.
- [ ] **Human review interface (review.py)** — UI to review flagged questions, approve/reject, edit. Mentioned early, never built.
- [ ] **Tiered difficulty calibration** — verify easy/medium/hard labels by comparing against PYQ difficulty distribution.

### Scale
- [ ] **Cost optimisation** — switch question generation to Haiku for drafting, Sonnet for final polish pass. Estimate: 5× cheaper than current batch mode.
- [ ] **Batch size scaling** — test 50-question batch sets. Currently tested up to 10.

---

## 💡 Ideas (not yet assessed)

- Mains answer writing practice (long-form, not MCQ)
- Mock test mode — 100 questions, 2-hour timer, UPSC paper format
- Explanation quality improvement — current explanations are 2–3 sentences; could be richer
- Hindi language support for CA ingestion and question generation
