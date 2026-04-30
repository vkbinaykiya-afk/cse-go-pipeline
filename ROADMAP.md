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
- [ ] **Improve precision** — statements sometimes claim slightly beyond what the extract explicitly states (e.g. Article cited not in extract, partial descriptions creating ambiguity, extract-to-statement misalignment across multiple extracts). Fix: tighten batch prompt to require each statement to be fully verifiable from its single assigned extract; reject any statement that requires inference across extracts. Also: checker flags CA events with future dates as unverifiable — add a date-awareness note to the generation prompt.
- [ ] **Assertion-reason logical validation** — micro Haiku call after generation to verify R causally explains A, not just correlates. Option: restrict AR to agentic mode only for now. ~1 day, ~$0.01/question.
- [ ] **Fix reword failures** — 34–47 chunks per PDF fall back to raw text. Likely Haiku timeouts on dense chunks. Add chunking pre-filter or retry logic.
- [ ] **Filter low-content CA chunks** — December 2025 PDF was a table of contents; chunks are index entries not substantive text. Add minimum word/sentence density filter before reword.

### Coverage
- [ ] **More CA months** — add remaining Vision IAS months (currently Dec 2025, Jan 2026, Feb 2026 only). Target: 12 months rolling.
- [ ] **PIB integration** — `ingest_ca.py --pib` is built but untested end-to-end. Test with 14-day pull.
- [ ] **Subject coverage audit** — run `agent_generate.py --subject` across all UPSC Prelims subjects, identify gaps in NCERT chunk retrieval.

### Pipeline
- [ ] **Guard daily set from mid-day overwrites** — `build_daily_set` currently uses `INSERT OR REPLACE`, so rerunning the pipeline mid-day resets the set. Once real users are active, skip set creation if today's set already exists AND has at least 1 attempt recorded against it. Safe for dev (no attempts = always overwrite), safe for prod (in-progress quiz is never reset).
- [ ] **DB as source of truth** — currently DB is rebuilt from JSON on every API restart; questions have no persistent identity, timestamps, or history. Refactor: `agent_generate`, `check`, `repair` write directly to SQLite. JSON becomes archive only. Add `generated_at`, `checked_at`, `repaired_at`, `pipeline_version` columns. Prerequisite for topic quizzes, archives, and recall.
- [ ] **`--subject` batch mode** — currently `--subject` only works in agentic mode. Wire planner to batch mode.
- [ ] **Daily question set automation** — cron/scheduler to generate N questions per day and write to `questions/daily/`.
- [x] **Question deduplication** — before saving, check new question against existing batches to avoid repeats on same topic.

---

## 🗂 Parked (discuss before building)

### Product
- [ ] **Lovable-based frontend** — user said defer. Needs: question display, answer reveal, score tracking, subject filter. API layer needed first.
- [ ] **API layer** — FastAPI or similar to serve questions and accept answers. Prerequisite for any frontend.
- [ ] **User progress tracking** — which questions attempted, weak subjects, accuracy over time.
- [ ] **Personalised practice mode** — retry questions answered wrong, drill weak topics. Pure SQL, no LLM. Needs two endpoints: `/questions/retry-wrongs` and `/questions/weak-areas`. Prerequisite: DB as source of truth + timestamps.

### Quality
- [ ] **Assertion-reason format (option 2)** — logical validation micro-call. Parked until pass rate is stable on statement_based.
- [ ] **Assertion-reason format (option 3)** — only generate AR when source text contains explicit causal language. More elegant but more work.
- [ ] **Human review interface (review.py)** — UI to review flagged questions, approve/reject, edit. Mentioned early, never built.
- [ ] **Tiered difficulty calibration** — verify easy/medium/hard labels by comparing against PYQ difficulty distribution.

### Scale
- [ ] **Cost optimisation** — switch question generation to Haiku for drafting, Sonnet for final polish pass. Estimate: 5× cheaper than current batch mode.
- [ ] **Batch size scaling** — test 50-question batch sets. Currently tested up to 10.

---

## 📖 Claude-Generated Wiki (assessed, not yet built)

### Goal
Replace raw NCERT chunk retrieval (RAG) with pre-distilled 500-word wiki articles per UPSC topic. Improves question groundedness, reduces drop rate from ~15% (post-full-coverage RAG) to ~8–12%.

### Why wiki beats RAG for question quality
- **Context fragmentation**: key facts sometimes span 2 chunks; generator only gets one → vague questions or unsupported answers
- **Noise**: raw NCERT text includes figure captions, exercise questions, headers mixed in with content
- **Retrieval unpredictability**: embedding similarity finds topically adjacent chunks, not always the factually relevant one
- Wiki articles are pre-synthesised (e.g. "Mauryan Empire" article has Chandragupta + Ashoka + administration + decline in one coherent piece), clean prose, and topic-indexed — not embedding-indexed

### Architecture

**Two-tier wiki**

| Tier | Content | Update frequency |
|------|---------|-----------------|
| Static | NCERT (Class 6–12), Old NCERTs, Environment, PYQs | Once — regenerate only if source books change |
| Dynamic | CA monthly PDFs, Economic Survey, new uploads | Per upload — uploader triggers article generation after ingest |

**Generation pipeline**
1. Cluster existing chunks by topic (using `upsc_subject` + `upsc_topic` tags already on chunks)
2. For each topic cluster: retrieve top-5 chunks → Haiku generates a 500-word structured article
3. Store articles in a new ChromaDB collection `cse_wiki`
4. Question generator queries `cse_wiki` instead of `cse_knowledge_base`

**Uploader integration (dynamic tier)**
After ingest → auto-generate wiki articles for newly added chunks → store in `cse_wiki` → R2 sync includes both collections

### Cost & effort

| Phase | Effort | Cost |
|-------|--------|------|
| Build generation script (`build_wiki.py`) | ~3–4 hrs | — |
| One-time static wiki generation (~900 articles) | ~25 min runtime | ~$7 (Haiku) |
| Per CA month upload (dynamic, ~50 articles) | ~2 min runtime | ~$0.35 |
| Switch question generator to use `cse_wiki` | ~1 hr | — |
| **Total setup** | **~1 afternoon** | **~$10** |

### Trigger condition
Build after full content coverage is reached (6 remaining CA months + bare acts). No point generating wiki over an incomplete corpus — static tier should be generated once over the final complete source set.

**Estimated effort**: 1 day. Prerequisite: full content coverage.

---

## 🗺 Map Questions (assessed, not yet built)

### Goal
Mandatory visual map aid for all `map_based` questions. Simplified abstraction — not satellite/photo — matching NCERT textbook style: clean outlines, minimal colour, only relevant entities labelled.

### Architecture

**Rendering stack**
- `generate_map.py` — Python script, takes a `map_config` JSON, outputs a PNG to `./static/maps/`
- Libraries: `geopandas` + `matplotlib` (deterministic, no API calls, same config = same map)
- Output stored as `image_url TEXT` on the question record (DB column + migration already planned)

**Two-tier feature data**

| Tier | Source | Coverage |
|------|--------|----------|
| Base layer | Natural Earth GeoJSON (free) | Major rivers, state/country boundaries, coastlines, mountain ranges |
| Strategic layer | GADM (district-level India) + hand-curated GeoJSON (~50 points) | Northeast state detail, LAC/LoC segments, passes (Doklam, Galwan, Nathu La, Siachen), straits (Hormuz, Bab-el-Mandeb, Gulf of Aden), corridors (INSTC, Chabahar, BRI routes), SCO member boundaries |

**CA-aware entity promotion**
When a geographic entity appears in CA ingestion → extracted and stored in a `geo_entities` table (name, lat/lon, bbox, type). At map question generation time, the LLM picks entities from `geo_entities` + base layer. Entities "in news" are automatically surfaced as label/highlight candidates.

**Generation prompt addition**
For `map_based` questions, generator outputs a `map_config` block alongside the question:
```json
{
  "bbox": [68.0, 8.0, 97.0, 37.0],
  "zoom_region": "peninsular india",
  "highlight": ["Krishna River", "Godavari River"],
  "label": ["Karnataka", "Andhra Pradesh", "Tamil Nadu"],
  "show_layers": ["states", "rivers"]
}
```

**Priority regions for UPSC**
- Northeast India — Brahmaputra tributaries, Seven Sisters boundaries, passes, Doklam plateau
- Border states — LAC (Western/Middle/Eastern sectors), LoC, Aksai Chin, Siachen
- Middle East — Strait of Hormuz, Bab-el-Mandeb, Red Sea corridor, Gulf of Aden, conflict zones
- Central Asia — SCO members, INSTC route, Chabahar port, BRI corridors

**Build order**
1. Add `map_config TEXT` + `image_url TEXT` columns to DB schema
2. Curate strategic GeoJSON (~50 points, ~2 hrs one-time task)
3. Build `generate_map.py` with geopandas
4. Add `map_config` extraction to generation prompt for `map_based` question type
5. Wire `geo_entities` extraction into CA ingestion pipeline
6. Wire `image_url` into frontend question display

**Estimated effort**: 3–4 days. Covers ~90% of UPSC map questions with major + strategically important minor features.

---

## 🎯 Quiz UX — Skip & Confidence (assessed, not yet built)

**Practice mode**
- Skip button: low-friction, lets candidate defer and revisit before finishing
- Show "skipped" indicator on question nav; allow revisit before submitting

**Exam mode** (future)
- Skip is a strategic decision — UPSC scoring: +2 correct, -0.66 wrong, 0 skipped
- Label the button "Skip (no penalty)" to reinforce exam strategy, not just a neutral skip
- Keyboard shortcut for skip (serious aspirants are keyboard-first)

**Confidence indicator — both modes**
Three-state answer: **Sure / Unsure / Skip**
- Sure: attempted, confident
- Unsure: attempted, flagged for review — show differently at summary
- Skip: not attempted this session

**Why this matters beyond UX:** Unsure rate per question is a quality signal. High Unsure rate = question is ambiguous or too hard → feeds back into HITL review queue. This turns user behaviour into a passive QA loop.

**Data to store:** add `confidence` column to `attempts` table (`sure | unsure | skip`). Aggregate in `/report` as a per-question and per-subject metric.

---

## 💡 Ideas (not yet assessed)

- Mains answer writing practice (long-form, not MCQ)
- Mock test mode — 100 questions, 2-hour timer, UPSC paper format
- Explanation quality improvement — current explanations are 2–3 sentences; could be richer
- Hindi language support for CA ingestion and question generation
