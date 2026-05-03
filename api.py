"""
FastAPI backend for CSE-GO quiz app.
Serves questions, records attempts, returns personalized reports.
Uses PostgreSQL on Railway (DATABASE_URL env var), SQLite locally.
"""

import json
import glob
import os
import uuid
import secrets
import string
import smtplib
import urllib.request
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from pathlib import Path
from email.mime.text import MIMEText

from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

GMAIL_USER        = os.environ.get("GMAIL_USER", "")
GMAIL_APP_PASSWORD= os.environ.get("GMAIL_APP_PASSWORD", "")
RESEND_API_KEY    = os.environ.get("RESEND_API_KEY", "")
RESEND_FROM       = os.environ.get("RESEND_FROM", "onboarding@resend.dev")
BREVO_API_KEY     = os.environ.get("BREVO_API_KEY", "")
OTP_EXPIRY_MINUTES= 10
PORT              = int(os.environ.get("PORT", 8000))
PIPELINE_SECRET   = os.environ.get("PIPELINE_SECRET", "")
DATABASE_URL      = os.environ.get("DATABASE_URL", "")   # set by Railway Postgres
STATIC_SUBJECTS   = ["History", "Art & Culture", "Geography", "Polity", "Environment",
                     "Science & Technology", "Economics", "Current Affairs"]

# Canonical subject name lookup — applied everywhere subjects are read from the DB.
_SUBJ_NORM_MAP = {
    "economy":                  "Economics",
    "science & tech":           "Science & Technology",
    "science and technology":   "Science & Technology",
    "sci & tech":               "Science & Technology",
    "s&t":                      "Science & Technology",
    "indian polity":            "Polity",
    "polity & governance":      "Polity",
    "polity and governance":    "Polity",
    "art & culture":            "Art & Culture",
    "arts & culture":           "Art & Culture",
    "current affairs":          "Current Affairs",
    "modern history":           "History",
    "ancient history":          "History",
    "medieval history":         "History",
    "indian history":           "History",
    "general studies":          "General",
    "gs":                       "General",
    "general knowledge":        "General",
}

def _norm_subject(s) -> str:
    if not s:
        return "General"
    return _SUBJ_NORM_MAP.get(s.strip().lower(), s.strip())

DB_PATH       = Path(__file__).parent / "cse_go.db"
QUESTIONS_DIR = Path(__file__).parent / "questions"

app = FastAPI(title="CSE-GO API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

USE_PG = bool(DATABASE_URL)

# ---------------------------------------------------------------------------
# DB connection — Postgres or SQLite
# ---------------------------------------------------------------------------

def get_db():
    if USE_PG:
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = False
        return conn
    else:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn


def _execute(conn, sql, params=()):
    """Execute SQL, handling ? vs %s placeholder difference."""
    if USE_PG:
        sql = sql.replace("?", "%s")
        # Replace SQLite-only syntax
        sql = sql.replace("INSERT OR REPLACE INTO", "INSERT INTO")
        sql = sql.replace("INSERT OR IGNORE INTO", "INSERT INTO")
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur


def _fetchone(cur, conn):
    row = cur.fetchone()
    if row is None:
        return None
    if USE_PG:
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))
    return row


def _fetchall(cur, conn):
    rows = cur.fetchall()
    if USE_PG:
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in rows]
    return rows


def _row_get(row, key, default=None):
    """Get value from either a dict (PG) or sqlite3.Row."""
    if row is None:
        return default
    try:
        return row[key]
    except (KeyError, IndexError):
        return default


def _commit(conn):
    if not USE_PG or not conn.autocommit:
        conn.commit()


# ---------------------------------------------------------------------------
# DB init
# ---------------------------------------------------------------------------

def init_db():
    conn = get_db()
    if USE_PG:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id                TEXT PRIMARY KEY,
                question          TEXT NOT NULL,
                options           TEXT NOT NULL,
                correct           TEXT NOT NULL,
                explanation       TEXT,
                subject           TEXT,
                difficulty        TEXT,
                question_type     TEXT,
                source_type       TEXT,
                source_file       TEXT,
                source_page       INTEGER,
                status            TEXT,
                flag_reason       TEXT,
                extracts          TEXT,
                raw               TEXT NOT NULL,
                upsc_subject      TEXT,
                upsc_topic        TEXT,
                broad_category    TEXT,
                question_category TEXT,
                generated_at      TEXT,
                checked_at        TEXT,
                repaired_at       TEXT,
                pipeline_version  TEXT,
                topic_query       TEXT,
                suggested_reading TEXT
            );
            CREATE TABLE IF NOT EXISTS attempts (
                id               TEXT PRIMARY KEY,
                question_id      TEXT NOT NULL,
                chosen           TEXT,
                is_correct       INTEGER,
                time_taken       INTEGER,
                attempted_at     TEXT NOT NULL,
                user_id          TEXT,
                is_daily         INTEGER NOT NULL DEFAULT 0,
                was_skipped      INTEGER DEFAULT 0,
                best_guess       TEXT,
                guess_correct    INTEGER,
                marks_actual     REAL,
                marks_intuition  REAL DEFAULT 0,
                quiz_session_id  TEXT
            );
            CREATE TABLE IF NOT EXISTS daily_sets (
                date         TEXT PRIMARY KEY,
                question_ids TEXT NOT NULL,
                created_at   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS users (
                id           TEXT PRIMARY KEY,
                email        TEXT UNIQUE NOT NULL,
                signup_date  TEXT NOT NULL,
                created_at   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS otps (
                email      TEXT PRIMARY KEY,
                code       TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token      TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS review_batches (
                date TEXT PRIMARY KEY,
                question_ids TEXT NOT NULL,
                staged_at TEXT NOT NULL,
                auto_publish_at TEXT NOT NULL,
                published_at TEXT,
                published_by TEXT,
                prompt_notes TEXT
            );
        """)
        conn.commit()
    else:
        import sqlite3
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS questions (
                id TEXT PRIMARY KEY, question TEXT NOT NULL, options TEXT NOT NULL,
                correct TEXT NOT NULL, explanation TEXT, subject TEXT, difficulty TEXT,
                question_type TEXT, source_type TEXT, source_file TEXT, source_page INTEGER,
                status TEXT, flag_reason TEXT, extracts TEXT, raw TEXT NOT NULL,
                upsc_subject TEXT, upsc_topic TEXT, broad_category TEXT,
                question_category TEXT, generated_at TEXT, checked_at TEXT,
                repaired_at TEXT, pipeline_version TEXT, topic_query TEXT, suggested_reading TEXT
            );
            CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY, question_id TEXT NOT NULL, chosen TEXT,
                is_correct INTEGER, time_taken INTEGER, attempted_at TEXT NOT NULL,
                user_id TEXT, is_daily INTEGER NOT NULL DEFAULT 0,
                was_skipped INTEGER DEFAULT 0, best_guess TEXT, guess_correct INTEGER,
                marks_actual REAL, marks_intuition REAL DEFAULT 0, quiz_session_id TEXT
            );
            CREATE TABLE IF NOT EXISTS daily_sets (
                date TEXT PRIMARY KEY, question_ids TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
                signup_date TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS otps (
                email TEXT PRIMARY KEY, code TEXT NOT NULL, expires_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS review_batches (
                date TEXT PRIMARY KEY,
                question_ids TEXT NOT NULL,
                staged_at TEXT NOT NULL,
                auto_publish_at TEXT NOT NULL,
                published_at TEXT,
                published_by TEXT,
                prompt_notes TEXT
            );
        """)
        conn.commit()
    conn.close()


def migrate_db():
    """Add new columns and repair dirty data (idempotent)."""
    new_cols = [
        ("attempts", "was_skipped",     "INTEGER DEFAULT 0"),
        ("attempts", "best_guess",       "TEXT"),
        ("attempts", "guess_correct",    "INTEGER"),
        ("attempts", "marks_actual",     "REAL"),
        ("attempts", "marks_intuition",  "REAL DEFAULT 0"),
        ("attempts", "quiz_session_id",  "TEXT"),
        ("questions", "review_decision", "TEXT DEFAULT 'accept'"),
    ]
    # Each column needs its own connection — psycopg2 puts the connection into
    # an aborted-transaction state on any error, blocking all subsequent commands
    # until an explicit ROLLBACK. Using a fresh connection per column avoids this.
    for table, col, defn in new_cols:
        conn = get_db()
        try:
            if USE_PG:
                cur = conn.cursor()
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {defn}")
                conn.commit()
            else:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {defn}")
                conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            conn.close()

    # Repair legacy skip rows: Lovable sent chosen="" before was_skipped existed.
    # These rows have chosen='' + is_correct=0 but should be skips (0 marks, not -0.66).
    conn = get_db()
    try:
        if USE_PG:
            cur = conn.cursor()
            cur.execute(
                "UPDATE attempts SET was_skipped=1, marks_actual=0, marks_intuition=0 "
                "WHERE (chosen='' OR chosen IS NULL) AND is_correct=0 AND was_skipped=0"
            )
            conn.commit()
        else:
            conn.execute(
                "UPDATE attempts SET was_skipped=1, marks_actual=0, marks_intuition=0 "
                "WHERE (chosen='' OR chosen IS NULL) AND is_correct=0 AND (was_skipped=0 OR was_skipped IS NULL)"
            )
            conn.commit()
    except Exception as e:
        print(f"[MIGRATE] legacy skip repair: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        conn.close()


def _marks_actual(is_correct: bool, was_skipped: bool) -> float:
    if was_skipped:
        return 0.0
    return 2.0 if is_correct else -0.66


def _marks_intuition(best_guess: Optional[str], guess_correct: Optional[bool]) -> float:
    if not best_guess:
        return 0.0
    return 2.0 if guess_correct else -0.66


def _upsert_otp(conn, email, code, expires_at):
    if USE_PG:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO otps (email, code, expires_at) VALUES (%s, %s, %s)
            ON CONFLICT (email) DO UPDATE SET code=EXCLUDED.code, expires_at=EXCLUDED.expires_at
        """, (email, code, expires_at))
    else:
        conn.execute("INSERT OR REPLACE INTO otps (email, code, expires_at) VALUES (?,?,?)",
                     (email, code, expires_at))


def _upsert_daily_set(conn, date, question_ids_json, created_at):
    if USE_PG:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO daily_sets (date, question_ids, created_at) VALUES (%s, %s, %s)
            ON CONFLICT (date) DO UPDATE SET question_ids=EXCLUDED.question_ids, created_at=EXCLUDED.created_at
        """, (date, question_ids_json, created_at))
    else:
        conn.execute("INSERT OR REPLACE INTO daily_sets (date, question_ids, created_at) VALUES (?,?,?)",
                     (date, question_ids_json, created_at))


# ---------------------------------------------------------------------------
# Question import
# ---------------------------------------------------------------------------

def sync_statuses():
    conn = get_db()
    files = glob.glob(str(QUESTIONS_DIR / "*.json"))
    updated = 0
    for fpath in files:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict):
            data = [data]
        for q in data:
            q_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, q.get("question", "") + q.get("source_file", "")))
            status = q.get("status")
            if status in ("pass", "flag"):
                cur = _execute(conn, "UPDATE questions SET status=?, flag_reason=? WHERE id=?",
                               (status, q.get("flag_reason"), q_id))
                updated += cur.rowcount
    _commit(conn)
    conn.close()
    return updated


def import_questions():
    conn = get_db()
    files = glob.glob(str(QUESTIONS_DIR / "*.json"))
    inserted = skipped = 0
    for fpath in files:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict):
            data = [data]
        for q in data:
            q_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, q.get("question", "") + q.get("source_file", "")))
            cur = _execute(conn, "SELECT id FROM questions WHERE id=?", (q_id,))
            if _fetchone(cur, conn):
                skipped += 1
                continue
            subject = _infer_subject(q)
            extracts = q.get("cited_extracts") or ([q["cited_extract"]] if q.get("cited_extract") else [])
            _execute(conn, """
                INSERT INTO questions
                  (id, question, options, correct, explanation, subject, difficulty,
                   question_type, source_type, source_file, source_page,
                   status, flag_reason, extracts, raw)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                q_id, q.get("question", ""), json.dumps(q.get("options", {})),
                q.get("correct_answer", ""), q.get("explanation", ""), subject,
                q.get("difficulty", "medium"), q.get("question_type", "statement_based"),
                q.get("source_type", "ncert"), q.get("source_file", ""),
                _safe_int(q.get("source_page")), q.get("status", "unchecked"),
                q.get("flag_reason"), json.dumps(extracts), json.dumps(q),
            ))
            inserted += 1
    _commit(conn)
    conn.close()
    return inserted, skipped


def _safe_int(val):
    if val is None:
        return None
    if isinstance(val, list):
        val = val[0] if val else None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _infer_subject(q: dict) -> str:
    src = (q.get("source_file") or q.get("topic_query") or "").lower()
    mapping = {
        "polity": "Polity", "political": "Polity", "constitution": "Polity",
        "geography": "Geography", "contemporary_india": "Geography",
        "history": "History", "economics": "Economics", "economy": "Economics",
        "environment": "Environment", "science": "Science & Technology",
        "biology": "Science & Technology", "physics": "Science & Technology",
        "chemistry": "Science & Technology", "current_affairs": "Current Affairs",
        "ramsar": "Environment",
    }
    for key, subject in mapping.items():
        if key in src:
            return subject
    if q.get("source_type") == "ca":
        return "Current Affairs"
    return "General Studies"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AttemptIn(BaseModel):
    question_id: str
    chosen: Optional[str] = None       # null when skipped
    time_taken: Optional[int] = None
    was_skipped: bool = False
    best_guess: Optional[str] = None   # A/B/C/D — only when was_skipped=True
    quiz_session_id: Optional[str] = None

class AttemptOut(BaseModel):
    id: str
    question_id: str
    chosen: Optional[str]
    is_correct: Optional[bool]
    correct_answer: str
    explanation: str
    was_skipped: bool
    best_guess: Optional[str]
    marks_actual: float
    marks_intuition: float

class OTPRequest(BaseModel):
    email: str

class OTPVerify(BaseModel):
    email: str
    code: str

class PushDailySetIn(BaseModel):
    date: str
    question_ids: List[str]
    secret: str

class PushQuestionsIn(BaseModel):
    questions: List[dict]
    secret: str
    update_status: bool = False

class StageBatchIn(BaseModel):
    date: str
    question_ids: List[str]
    secret: str

class ReviewPublishIn(BaseModel):
    date: str
    prompt_notes: Optional[str] = None
    secret: str

class ReviewQuestionUpdateIn(BaseModel):
    decision: Optional[str] = None
    upsc_subject: Optional[str] = None
    question: Optional[str] = None
    options: Optional[dict] = None
    explanation: Optional[str] = None
    secret: str


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _get_user_from_token(token: Optional[str], conn) -> Optional[dict]:
    if not token:
        return None
    cur = _execute(conn,
        "SELECT u.id, u.email, u.signup_date FROM sessions s "
        "JOIN users u ON s.user_id = u.id WHERE s.token = ?", (token,))
    row = _fetchone(cur, conn)
    return dict(row) if row else None


def _send_otp_email(email: str, code: str):
    body = f"Your CSE-GO one-time login code is: {code}\n\nValid for {OTP_EXPIRY_MINUTES} minutes."

    if BREVO_API_KEY:
        import urllib.error
        payload = json.dumps({
            "sender": {"name": "CSE-GO", "email": "v.k.binaykiya@gmail.com"},
            "to": [{"email": email}],
            "subject": "Your CSE-GO login code",
            "textContent": body,
        }).encode()
        req = urllib.request.Request(
            "https://api.brevo.com/v3/smtp/email",
            data=payload,
            headers={"api-key": BREVO_API_KEY, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                pass
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Brevo {e.code}: {e.read().decode()}")
        return

    if RESEND_API_KEY:
        import urllib.error
        payload = json.dumps({
            "from": RESEND_FROM,
            "to": [email],
            "subject": "Your CSE-GO login code",
            "text": body,
        }).encode()
        req = urllib.request.Request(
            "https://api.resend.com/emails",
            data=payload,
            headers={"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                pass
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Resend {e.code}: {e.read().decode()}")
        return

    msg = MIMEText(body)
    msg["Subject"] = "Your CSE-GO login code"
    msg["From"] = GMAIL_USER
    msg["To"] = email
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as s:
        s.starttls()
        s.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        s.send_message(msg)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.get("/ping")
def ping():
    return {"ok": True}


@app.on_event("startup")
def startup():
    init_db()
    migrate_db()
    inserted, skipped = import_questions()
    updated = sync_statuses()
    print(f"DB ready ({'postgres' if USE_PG else 'sqlite'}). "
          f"Imported {inserted} new, {skipped} existing, {updated} statuses synced.")


def _topup_from_pool(conn, selected_ids: list, date: str, target: int = 10) -> list:
    needed = target - len(selected_ids)
    if needed <= 0:
        return selected_ids[:target]
    exclude = list(set(selected_ids))
    cur = _execute(conn, "SELECT question_ids FROM daily_sets ORDER BY date DESC LIMIT 7")
    for row in _fetchall(cur, conn):
        raw = row["question_ids"]
        exclude.extend(json.loads(raw) if isinstance(raw, str) else raw)
    exclude = list(set(exclude))

    def _excl(excl):
        if not excl:
            return "", ()
        ph = ",".join(["%s" if USE_PG else "?"] * len(excl))
        return f"AND id NOT IN ({ph})", tuple(excl)

    ec, ep = _excl(exclude)
    cur = _execute(conn,
        f"SELECT id FROM questions WHERE status='pass' "
        f"AND (source_type='pyq' OR LOWER(COALESCE(source_file,'')) LIKE '%pyq%') "
        f"{ec} LIMIT ?", ep + (needed,))
    selected_ids = selected_ids + [r["id"] for r in _fetchall(cur, conn)]
    if len(selected_ids) < target:
        still = target - len(selected_ids)
        ec2, ep2 = _excl(list(set(exclude + selected_ids)))
        cur = _execute(conn,
            f"SELECT id FROM questions WHERE status='pass' {ec2} ORDER BY RANDOM() LIMIT ?",
            ep2 + (still,))
        selected_ids += [r["id"] for r in _fetchall(cur, conn)]
    return selected_ids[:target]


def _check_and_auto_publish(conn):
    now_iso = datetime.now(timezone.utc).isoformat()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cur = _execute(conn,
        "SELECT date, question_ids FROM review_batches "
        "WHERE published_at IS NULL AND auto_publish_at <= ? AND date <= ?",
        (now_iso, today))
    overdue = _fetchall(cur, conn)
    for batch in overdue:
        date = batch["date"]
        raw = batch["question_ids"]
        q_ids = json.loads(raw) if isinstance(raw, str) else raw
        cur2 = _execute(conn, "SELECT date FROM daily_sets WHERE date=?", (date,))
        if _fetchone(cur2, conn):
            _execute(conn, "UPDATE review_batches SET published_at=?, published_by='auto' WHERE date=?",
                     (now_iso, date))
            continue
        ph = ",".join(["%s" if USE_PG else "?"] * len(q_ids))
        cur2 = _execute(conn,
            f"SELECT id FROM questions WHERE id IN ({ph}) AND COALESCE(review_decision,'accept') != 'reject'",
            tuple(q_ids))
        accepted = [r["id"] for r in _fetchall(cur2, conn)]
        selected = _topup_from_pool(conn, accepted[:10], date)
        if selected:
            _upsert_daily_set(conn, date, json.dumps(selected), now_iso)
        _execute(conn, "UPDATE review_batches SET published_at=?, published_by='auto' WHERE date=?",
                 (now_iso, date))
        print(f"[AUTO-PUBLISH] {date}: {len(selected)} questions auto-published")
    if overdue:
        _commit(conn)


# ---------------------------------------------------------------------------
# Routes — Questions
# ---------------------------------------------------------------------------

@app.get("/questions")
def list_questions(
    subject: Optional[str] = None, upsc_subject: Optional[str] = None,
    upsc_topic: Optional[str] = None, question_category: Optional[str] = None,
    difficulty: Optional[str] = None, question_type: Optional[str] = None,
    status: Optional[str] = Query(default="pass"), limit: int = 10, offset: int = 0,
):
    conn = get_db()
    clauses, params = [], []
    if subject:        clauses.append("subject = ?");          params.append(subject)
    if upsc_subject:   clauses.append("upsc_subject = ?");     params.append(upsc_subject)
    if upsc_topic:     clauses.append("upsc_topic = ?");       params.append(upsc_topic)
    if question_category: clauses.append("question_category = ?"); params.append(question_category)
    if difficulty:     clauses.append("difficulty = ?");       params.append(difficulty)
    if question_type:  clauses.append("question_type = ?");    params.append(question_type)
    if status and status != "all":
        clauses.append("status = ?"); params.append(status)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    order = "ORDER BY RANDOM()" if not USE_PG else "ORDER BY RANDOM()"
    cur = _execute(conn, f"SELECT * FROM questions {where} {order} LIMIT ? OFFSET ?", params + [limit, offset])
    rows = _fetchall(cur, conn)
    conn.close()
    return [_format_question(r) for r in rows]


@app.get("/questions/{question_id}")
def get_question(question_id: str):
    conn = get_db()
    cur = _execute(conn, "SELECT * FROM questions WHERE id=?", (question_id,))
    row = _fetchone(cur, conn)
    conn.close()
    if not row:
        raise HTTPException(404, "Question not found")
    return _format_question(row)


# ---------------------------------------------------------------------------
# Routes — Auth
# ---------------------------------------------------------------------------

@app.post("/auth/request-otp")
def request_otp(body: OTPRequest):
    code = "".join(secrets.choice(string.digits) for _ in range(6))
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRY_MINUTES)).isoformat()
    conn = get_db()
    _upsert_otp(conn, body.email.lower(), code, expires_at)
    _commit(conn)
    conn.close()
    if BREVO_API_KEY or RESEND_API_KEY or (GMAIL_USER and GMAIL_APP_PASSWORD):
        try:
            _send_otp_email(body.email.lower(), code)
        except Exception as e:
            print(f"[EMAIL ERROR] {e} — OTP for {body.email.lower()}: {code}")
    else:
        print(f"[DEV] OTP for {body.email}: {code}")
    return {"message": "OTP sent"}


@app.post("/auth/verify-otp")
def verify_otp(body: OTPVerify):
    conn = get_db()
    email = body.email.lower()
    cur = _execute(conn, "SELECT code, expires_at FROM otps WHERE email=?", (email,))
    row = _fetchone(cur, conn)
    if not row:
        conn.close()
        raise HTTPException(400, "No OTP found for this email")
    if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
        conn.close()
        raise HTTPException(400, "OTP expired")
    if row["code"] != body.code:
        conn.close()
        raise HTTPException(400, "Invalid OTP")

    cur = _execute(conn, "SELECT id, signup_date FROM users WHERE email=?", (email,))
    user = _fetchone(cur, conn)
    if not user:
        user_id = str(uuid.uuid4())
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        _execute(conn, "INSERT INTO users (id, email, signup_date, created_at) VALUES (?,?,?,?)",
                 (user_id, email, today, datetime.now(timezone.utc).isoformat()))
        signup_date = today
    else:
        user_id = user["id"]
        signup_date = user["signup_date"]

    token = secrets.token_urlsafe(32)
    _execute(conn, "INSERT INTO sessions (token, user_id, created_at) VALUES (?,?,?)",
             (token, user_id, datetime.now(timezone.utc).isoformat()))
    _execute(conn, "DELETE FROM otps WHERE email=?", (email,))
    _commit(conn)
    conn.close()
    return {"session_token": token, "user_id": user_id, "email": email, "signup_date": signup_date}


@app.post("/auth/logout")
def logout(x_session_token: Optional[str] = Header(default=None)):
    if x_session_token:
        conn = get_db()
        _execute(conn, "DELETE FROM sessions WHERE token=?", (x_session_token,))
        _commit(conn)
        conn.close()
    return {"message": "Logged out"}


@app.get("/auth/me")
def get_me(x_session_token: Optional[str] = Header(default=None)):
    conn = get_db()
    user = _get_user_from_token(x_session_token, conn)
    conn.close()
    if not user:
        raise HTTPException(401, "Not authenticated")
    return user


# ---------------------------------------------------------------------------
# Routes — Attempts
# ---------------------------------------------------------------------------

@app.post("/attempts", response_model=AttemptOut)
def record_attempt(body: AttemptIn, x_session_token: Optional[str] = Header(default=None)):
    import traceback as _tb
    conn = get_db()
    user = _get_user_from_token(x_session_token, conn)
    cur = _execute(conn, "SELECT * FROM questions WHERE id=?", (body.question_id,))
    q = _fetchone(cur, conn)
    if not q:
        conn.close()
        raise HTTPException(404, "Question not found")

    is_correct: Optional[bool] = None
    guess_correct: Optional[bool] = None

    # Treat chosen="" as a skip (Lovable backward-compat — sent "" before was_skipped existed)
    try:
        is_skip = body.was_skipped or not (body.chosen or "").strip()

        if is_skip:
            is_correct = None
            if body.best_guess and body.best_guess.strip():
                guess_correct = body.best_guess.upper() == q["correct"].upper()
        else:
            is_correct = body.chosen.upper() == q["correct"].upper()

        marks_a = _marks_actual(bool(is_correct), is_skip)
        marks_i = _marks_intuition(body.best_guess, guess_correct)

        attempt_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        today_utc = datetime.now(timezone.utc).date()
        yesterday_utc = (today_utc - timedelta(days=1)).strftime("%Y-%m-%d")
        today_str = today_utc.strftime("%Y-%m-%d")
        ph2 = "%s,%s" if USE_PG else "?,?"
        cur = _execute(conn, f"SELECT question_ids FROM daily_sets WHERE date IN ({ph2})",
                       (today_str, yesterday_utc))
        daily_ids: set = set()
        for drow in _fetchall(cur, conn):
            qids_raw = drow["question_ids"] if isinstance(drow, dict) else drow["question_ids"]
            daily_ids.update(json.loads(qids_raw) if isinstance(qids_raw, str) else qids_raw)
        is_daily = 1 if body.question_id in daily_ids else 0

        _execute(conn,
            "INSERT INTO attempts (id, question_id, chosen, is_correct, time_taken, attempted_at, "
            "user_id, is_daily, was_skipped, best_guess, guess_correct, marks_actual, marks_intuition, quiz_session_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (attempt_id, body.question_id, body.chosen if not is_skip else "",
             int(is_correct) if is_correct is not None else 0,
             body.time_taken, now, user["id"] if user else None, is_daily,
             int(is_skip), body.best_guess,
             int(guess_correct) if guess_correct is not None else None,
             marks_a, marks_i, body.quiz_session_id))
        _commit(conn)
        conn.close()

        return AttemptOut(
            id=attempt_id, question_id=body.question_id,
            chosen=body.chosen if not is_skip else None, is_correct=is_correct,
            correct_answer=q["correct"], explanation=q["explanation"] or "",
            was_skipped=is_skip, best_guess=body.best_guess,
            marks_actual=marks_a, marks_intuition=marks_i,
        )
    except HTTPException:
        raise
    except Exception as exc:
        try:
            conn.close()
        except Exception:
            pass
        print(f"[ATTEMPT ERROR] body={body!r} exc={exc!r}\n{_tb.format_exc()}")
        raise HTTPException(500, detail=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Routes — Debug
# ---------------------------------------------------------------------------

@app.get("/debug/skips")
def debug_skips(x_session_token: Optional[str] = Header(default=None)):
    """Return last 20 skip attempts with best_guess data — diagnostic only."""
    conn = get_db()
    user = _get_user_from_token(x_session_token, conn)
    if not user:
        conn.close()
        return {"error": "not authenticated"}
    uid = user["id"]
    if USE_PG:
        cur = conn.cursor()
        cur.execute("""
            SELECT a.question_id, a.was_skipped, a.best_guess, a.guess_correct,
                   a.marks_intuition, a.marks_actual, a.is_daily, a.attempted_at
            FROM attempts a
            WHERE a.user_id = %s AND a.was_skipped = 1
            ORDER BY a.attempted_at DESC LIMIT 20
        """, (uid,))
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    else:
        cur = _execute(conn, """
            SELECT question_id, was_skipped, best_guess, guess_correct,
                   marks_intuition, marks_actual, is_daily, attempted_at
            FROM attempts WHERE user_id = ? AND was_skipped = 1
            ORDER BY attempted_at DESC LIMIT 20
        """, (uid,))
        rows = [dict(r) for r in _fetchall(cur, conn)]
    conn.close()
    return {"user_id": uid, "skip_attempts": rows}


@app.get("/debug/daily-attempts")
def debug_daily_attempts(x_session_token: Optional[str] = Header(default=None)):
    """Show all attempts for today's daily set questions — no is_daily filter — diagnostic only."""
    conn = get_db()
    user = _get_user_from_token(x_session_token, conn)
    if not user:
        conn.close()
        return {"error": "not authenticated"}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cur = _execute(conn, "SELECT question_ids FROM daily_sets WHERE date=?", (today,))
    row = _fetchone(cur, conn)
    if not row:
        conn.close()
        return {"error": "no daily set for today", "today": today}
    q_ids = json.loads(row["question_ids"]) if isinstance(row["question_ids"], str) else row["question_ids"]
    ph = ",".join(["%s" if USE_PG else "?"] * len(q_ids))
    if USE_PG:
        cur = conn.cursor()
        cur.execute(f"SELECT question_id, chosen, is_correct, was_skipped, is_daily, best_guess, marks_intuition, attempted_at FROM attempts WHERE question_id IN ({ph}) AND user_id = %s ORDER BY attempted_at DESC", tuple(q_ids) + (user["id"],))
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    else:
        cur = _execute(conn, f"SELECT question_id, chosen, is_correct, was_skipped, is_daily, best_guess, marks_intuition, attempted_at FROM attempts WHERE question_id IN ({ph}) AND user_id=? ORDER BY attempted_at DESC", tuple(q_ids) + (user["id"],))
        rows = [dict(r) for r in _fetchall(cur, conn)]
    conn.close()
    return {"user_id": user["id"], "daily_set_date": today, "total_q_ids": len(q_ids), "attempts": rows}


# ---------------------------------------------------------------------------
# Routes — Quiz Score
# ---------------------------------------------------------------------------

@app.get("/quiz/score/{date}")
def get_quiz_score(date: str, x_session_token: Optional[str] = Header(default=None)):
    """UPSC-style score + intuition barometer for one quiz date."""
    conn = get_db()
    user = _get_user_from_token(x_session_token, conn)

    cur = _execute(conn, "SELECT question_ids FROM daily_sets WHERE date=?", (date,))
    row = _fetchone(cur, conn)
    if not row:
        conn.close()
        raise HTTPException(404, f"No daily set for {date}")

    q_ids = json.loads(row["question_ids"])
    total_questions = len(q_ids)
    max_marks = float(total_questions * 2)

    uid = user["id"] if user else None
    user_filter = "AND user_id = ?" if uid else ""
    user_p = [uid] if uid else []
    placeholders = ",".join(["?" if not USE_PG else "%s"] * len(q_ids))

    if USE_PG:
        cur2 = conn.cursor()
        ph = ",".join(["%s"] * len(q_ids))
        cur2.execute(f"""
            SELECT DISTINCT ON (a.question_id)
                a.question_id, a.chosen, a.is_correct, a.was_skipped,
                a.best_guess, a.guess_correct, a.marks_actual, a.marks_intuition,
                COALESCE(q.upsc_subject, q.subject, 'General') AS subject
            FROM attempts a
            JOIN questions q ON a.question_id = q.id
            WHERE a.question_id IN ({ph}) {user_filter.replace('?','%s')}
            ORDER BY a.question_id, a.attempted_at DESC
        """, tuple(q_ids) + tuple(user_p))
        cols = [d[0] for d in cur2.description]
        attempt_rows = [dict(zip(cols, r)) for r in cur2.fetchall()]
    else:
        cur = _execute(conn, f"""
            SELECT a.question_id, a.chosen, a.is_correct, a.was_skipped,
                   a.best_guess, a.guess_correct, a.marks_actual, a.marks_intuition,
                   COALESCE(q.upsc_subject, q.subject, 'General') AS subject
            FROM attempts a
            JOIN questions q ON a.question_id = q.id
            WHERE a.question_id IN ({placeholders}) {user_filter}
            GROUP BY a.question_id HAVING a.attempted_at = MAX(a.attempted_at)
        """, q_ids + user_p)
        attempt_rows = _fetchall(cur, conn)

    amap = {r["question_id"]: r for r in attempt_rows}

    attempted = correct = wrong = skipped = guessed = guess_correct_n = 0
    total_marks = total_intuition = 0.0
    subj_actual: dict = {}
    subj_intuition: dict = {}

    for q_id in q_ids:
        a = amap.get(q_id)
        if not a:
            continue

        ws = bool(a.get("was_skipped"))
        ma = a.get("marks_actual")
        mi = float(a.get("marks_intuition") or 0.0)
        subj = _norm_subject(a.get("subject") or "General")

        # Back-fill marks for legacy rows (pre-migration attempts).
        # If chosen is empty and is_correct is falsy, it was a skip recorded before
        # was_skipped existed — treat as 0, not -0.66.
        if ma is None:
            chosen_val = (a.get("chosen") or "").strip()
            if not chosen_val and not a.get("is_correct"):
                ma = 0.0   # legacy skip
            else:
                ma = _marks_actual(bool(a.get("is_correct")), ws)

        if ws:
            skipped += 1
            if a.get("best_guess"):
                guessed += 1
                if a.get("guess_correct"):
                    guess_correct_n += 1
        else:
            attempted += 1
            if a.get("is_correct"):
                correct += 1
            else:
                wrong += 1

        total_marks += ma
        total_intuition += mi

        subj_actual.setdefault(subj, {"marks": 0.0, "attempted": 0, "wrong": 0, "skipped": 0})
        subj_actual[subj]["marks"] += ma
        if ws:
            subj_actual[subj]["skipped"] += 1
        elif a.get("is_correct"):
            subj_actual[subj]["attempted"] += 1
        else:
            subj_actual[subj]["attempted"] += 1
            subj_actual[subj]["wrong"] += 1

        subj_intuition.setdefault(subj, {"skipped": 0, "guessed": 0, "guess_correct": 0, "intuition_marks": 0.0})
        if ws:
            subj_intuition[subj]["skipped"] += 1
            if a.get("best_guess"):
                subj_intuition[subj]["guessed"] += 1
                if a.get("guess_correct"):
                    subj_intuition[subj]["guess_correct"] += 1
            subj_intuition[subj]["intuition_marks"] += mi

    score_pct = round(total_marks / max_marks * 100, 1) if max_marks else 0
    adjusted_pct = round((total_marks + total_intuition) / max_marks * 100, 1) if max_marks else 0
    delta_pct = round(total_intuition / max_marks * 100, 1) if max_marks else 0

    if total_intuition > 0.5:
        intuition_msg = "You could take more risk — your gut would have lifted your score"
    elif total_intuition < -0.5:
        intuition_msg = "Smart skips — attempting those would have cost you marks"
    else:
        intuition_msg = "Coin-flip territory — skipping was the right call"

    subj_breakdown = [
        {"subject": s, "marks": round(v["marks"], 2),
         "attempted": v["attempted"], "wrong": v["wrong"], "skipped": v["skipped"]}
        for s, v in subj_actual.items()
    ]
    intuition_breakdown = [
        {"subject": s, "skipped": v["skipped"], "guessed": v["guessed"],
         "guess_correct": v["guess_correct"],
         "intuition_marks": round(v["intuition_marks"], 2),
         "intuition_accuracy_pct": round(v["guess_correct"] / v["guessed"] * 100, 1) if v["guessed"] else None}
        for s, v in subj_intuition.items() if v["skipped"] > 0
    ]

    conn.close()
    return {
        "date": date,
        "total_questions": total_questions,
        "attempted": attempted, "correct": correct, "wrong": wrong,
        "skipped": skipped, "unanswered": total_questions - attempted - skipped,
        "marks_actual": round(total_marks, 2),
        "marks_max": max_marks,
        "score_pct": score_pct,
        "guessed": guessed,
        "guess_correct": guess_correct_n,
        "guess_wrong": guessed - guess_correct_n,
        "intuition_marks": round(total_intuition, 2),
        "adjusted_marks": round(total_marks + total_intuition, 2),
        "adjusted_pct": adjusted_pct,
        "delta_pct": delta_pct,
        "intuition_message": intuition_msg,
        "subject_breakdown": subj_breakdown,
        "intuition_breakdown": intuition_breakdown,
    }


# ---------------------------------------------------------------------------
# Routes — Report
# ---------------------------------------------------------------------------

@app.get("/report")
def get_report(x_session_token: Optional[str] = Header(default=None)):
    conn = get_db()
    user = _get_user_from_token(x_session_token, conn)
    empty = {
        "total_attempts": 0, "overall_accuracy_pct": 0, "streak_days": 0,
        "subject_breakdown": [
            {"subject": s, "total": 0, "correct": 0, "accuracy_pct": None, "status": "not_attempted"}
            for s in STATIC_SUBJECTS
        ],
        "weak_areas": [],
    }
    if not user:
        conn.close()
        return empty

    uid = user["id"]
    try:
        cur = _execute(conn, "SELECT COUNT(*) FROM attempts WHERE user_id = ?", (uid,))
        total_attempts = _fetchone(cur, conn)
        total_attempts = list(total_attempts.values())[0] if USE_PG else total_attempts[0]
    except Exception as e:
        print(f"[REPORT ERROR - total_attempts] {e}")
        conn.close(); return empty

    try:
        cur = _execute(conn, "SELECT COUNT(*) FROM attempts WHERE user_id = ? AND is_correct=1", (uid,))
        correct_attempts = _fetchone(cur, conn)
        correct_attempts = list(correct_attempts.values())[0] if USE_PG else correct_attempts[0]
    except Exception as e:
        print(f"[REPORT ERROR - correct_attempts] {e}")
        conn.close(); return empty

    try:
        # Unified query — counts non-skipped attempts only for accuracy denominator
        cur = _execute(conn, """
            SELECT COALESCE(q.upsc_subject, q.subject, 'Unknown') AS subject,
                   COUNT(a.id) AS total,
                   SUM(CASE WHEN a.is_correct = 1 THEN 1 ELSE 0 END) AS correct,
                   SUM(CASE WHEN a.was_skipped = 1 THEN 1 ELSE 0 END) AS skipped_n,
                   COALESCE(SUM(
                       CASE
                           WHEN a.marks_actual IS NOT NULL THEN a.marks_actual
                           WHEN a.was_skipped = 1         THEN 0.0
                           WHEN a.is_correct  = 1         THEN 2.0
                           WHEN a.is_correct  = 0         THEN -0.66
                           ELSE 0.0
                       END
                   ), 0) AS marks_sum
            FROM attempts a JOIN questions q ON a.question_id = q.id
            WHERE a.user_id = ? GROUP BY COALESCE(q.upsc_subject, q.subject, 'Unknown')
        """, (uid,))
        raw_rows = _fetchall(cur, conn)
        attempted_map = {}
        for r in raw_rows:
            key = _norm_subject(r["subject"] or "")
            non_skip = int(r["total"] or 0) - int(r["skipped_n"] or 0)  # correct + wrong only
            correct_n = int(r["correct"] or 0)
            marks_n = float(r["marks_sum"] or 0)
            if key in attempted_map:
                prev = attempted_map[key]
                prev["total"] += non_skip
                prev["correct"] += correct_n
                prev["marks"] = round(prev["marks"] + marks_n, 2)
            else:
                attempted_map[key] = {
                    "subject": key, "total": non_skip,
                    "correct": correct_n, "marks": round(marks_n, 2),
                }
        for v in attempted_map.values():
            v["accuracy_pct"] = round(v["correct"] / v["total"] * 100, 1) if v["total"] > 0 else None
    except Exception as e:
        print(f"[REPORT ERROR - subject_breakdown] {e}")
        attempted_map = {}

    subject_breakdown = []
    for subj in STATIC_SUBJECTS:
        if subj in attempted_map:
            r = attempted_map[subj]
            subject_breakdown.append({
                "subject": subj, "total": r["total"], "correct": r["correct"],
                "accuracy_pct": r["accuracy_pct"],
                "marks": r["marks"],
                "status": "attempted" if r["total"] > 0 else "not_attempted",
            })
        else:
            subject_breakdown.append({
                "subject": subj, "total": 0, "correct": 0,
                "accuracy_pct": None, "marks": None, "status": "not_attempted",
            })

    weak_areas = sorted(
        [s for s in subject_breakdown if s["status"] == "attempted" and s["total"] >= 2
         and s["accuracy_pct"] is not None],
        key=lambda s: s["accuracy_pct"]
    )[:3]

    # Streak — consecutive completed days ending at yesterday (today excluded).
    # A "completed" day = all questions in that day's set were attempted.
    # Today is excluded so missing yesterday always shows 0, not 1.
    streak = 0
    try:
        today_utc = datetime.now(timezone.utc).date()
        if USE_PG:
            cur = conn.cursor()
            cur.execute("""
                SELECT ds.date, ds.question_ids,
                       COUNT(DISTINCT a.question_id) AS answered
                FROM daily_sets ds
                LEFT JOIN attempts a
                  ON a.user_id = %s
                  AND DATE(a.attempted_at::timestamptz) = ds.date::date
                GROUP BY ds.date, ds.question_ids
                ORDER BY ds.date DESC
            """, (uid,))
            completed_set = {
                str(r[0]) for r in cur.fetchall()
                if r[2] >= len(json.loads(r[1]))
                   and str(r[0]) != str(today_utc)
            }
        else:
            cur = _execute(conn, """
                SELECT ds.date,
                       (SELECT COUNT(DISTINCT a.question_id) FROM attempts a
                        WHERE a.question_id IN (SELECT value FROM json_each(ds.question_ids))
                        AND DATE(a.attempted_at) = ds.date AND a.user_id = ?) AS answered,
                       json_array_length(ds.question_ids) AS total
                FROM daily_sets ds ORDER BY ds.date DESC
            """, (uid,))
            completed_set = {
                str(r["date"]) for r in _fetchall(cur, conn)
                if (r["answered"] or 0) >= (r["total"] or 1)
                   and str(r["date"]) != str(today_utc)
            }

        check = today_utc - timedelta(days=1)
        while str(check) in completed_set:
            streak += 1
            check -= timedelta(days=1)
    except Exception as e:
        print(f"[STREAK ERROR] {e}")

    # Cumulative UPSC marks + intuition.
    # Legacy attempts pre-date the marks_actual column and have NULL there —
    # fall back to computing from is_correct / was_skipped.
    _MARKS_EXPR = """
        COALESCE(SUM(
            CASE
                WHEN marks_actual IS NOT NULL THEN marks_actual
                WHEN was_skipped = 1          THEN 0.0
                WHEN is_correct  = 1          THEN 2.0
                WHEN is_correct  = 0          THEN -0.66
                ELSE 0.0
            END
        ), 0)
    """
    _INTUITION_EXPR = "COALESCE(SUM(COALESCE(marks_intuition, 0)), 0)"
    try:
        if USE_PG:
            cur = conn.cursor()
            cur.execute(f"SELECT {_MARKS_EXPR}, {_INTUITION_EXPR} FROM attempts WHERE user_id=%s", (uid,))
            r = cur.fetchone()
            total_marks_sum, total_intuition_sum = float(r[0]), float(r[1])
        else:
            cur = _execute(conn, f"SELECT {_MARKS_EXPR}, {_INTUITION_EXPR} FROM attempts WHERE user_id=?", (uid,))
            r = _fetchone(cur, conn); total_marks_sum = float(r[0]); total_intuition_sum = float(r[1])
        max_marks_possible = total_attempts * 2.0
        score_pct = round(total_marks_sum / max_marks_possible * 100, 1) if max_marks_possible else 0
        delta_pct = round(total_intuition_sum / max_marks_possible * 100, 1) if max_marks_possible else 0
        adjusted_pct = round((total_marks_sum + total_intuition_sum) / max_marks_possible * 100, 1) if max_marks_possible else 0
    except Exception as e:
        print(f"[REPORT ERROR - marks] {e}")
        total_marks_sum = total_intuition_sum = score_pct = delta_pct = adjusted_pct = 0

    # ---- quiz_history: per-quiz scores for last 15 attempted daily sets ----
    quiz_history = []
    try:
        if USE_PG:
            cur2 = conn.cursor()
            cur2.execute("""
                SELECT ds.date, ds.question_ids
                FROM daily_sets ds
                WHERE EXISTS (
                    SELECT 1 FROM attempts a
                    WHERE a.user_id = %s
                    AND a.question_id = ANY(
                        SELECT json_array_elements_text(ds.question_ids::json)
                    )
                )
                ORDER BY ds.date DESC LIMIT 15
            """, (uid,))
            date_rows = [{"date": str(r[0]), "question_ids": r[1]} for r in cur2.fetchall()]
        else:
            cur = _execute(conn, """
                SELECT ds.date, ds.question_ids FROM daily_sets ds
                WHERE EXISTS (
                    SELECT 1 FROM attempts a WHERE a.user_id = ?
                    AND a.question_id IN (SELECT value FROM json_each(ds.question_ids))
                )
                ORDER BY ds.date DESC LIMIT 15
            """, (uid,))
            date_rows = _fetchall(cur, conn)

        for drow in date_rows:
            d_date = drow["date"]
            q_ids_raw = drow["question_ids"]
            q_ids_d = json.loads(q_ids_raw) if isinstance(q_ids_raw, str) else q_ids_raw
            if not q_ids_d:
                continue
            ph = ",".join(["%s" if USE_PG else "?"] * len(q_ids_d))
            if USE_PG:
                cur3 = conn.cursor()
                cur3.execute(f"""
                    SELECT DISTINCT ON (a.question_id)
                        a.is_correct, a.was_skipped, a.marks_actual, a.marks_intuition,
                        a.best_guess, a.guess_correct,
                        COALESCE(q.upsc_subject, q.subject, 'General') AS subject
                    FROM attempts a JOIN questions q ON a.question_id = q.id
                    WHERE a.question_id IN ({ph}) AND a.user_id = %s
                    ORDER BY a.question_id, a.attempted_at DESC
                """, tuple(q_ids_d) + (uid,))
                cols3 = [desc[0] for desc in cur3.description]
                att_rows = [dict(zip(cols3, r)) for r in cur3.fetchall()]
            else:
                cur = _execute(conn, f"""
                    SELECT a.is_correct, a.was_skipped, a.marks_actual, a.marks_intuition,
                           a.best_guess, a.guess_correct,
                           COALESCE(q.upsc_subject, q.subject, 'General') AS subject
                    FROM attempts a JOIN questions q ON a.question_id = q.id
                    WHERE a.question_id IN ({ph}) AND a.user_id = ?
                    GROUP BY a.question_id HAVING a.attempted_at = MAX(a.attempted_at)
                """, q_ids_d + [uid])
                att_rows = _fetchall(cur, conn)

            max_m = len(q_ids_d) * 2.0
            q_correct = q_wrong = q_skipped = q_guessed = q_gc = 0
            q_marks = q_intuition = 0.0
            subj_m: dict = {}

            for a in att_rows:
                ws = bool(a["was_skipped"] if isinstance(a, dict) else a["was_skipped"])
                ma = a["marks_actual"] if isinstance(a, dict) else a["marks_actual"]
                mi = float((a["marks_intuition"] if isinstance(a, dict) else a["marks_intuition"]) or 0)
                subj = _norm_subject((a["subject"] if isinstance(a, dict) else a["subject"]) or "General")
                if ma is None:
                    ma = 0.0 if ws else _marks_actual(bool(a["is_correct"] if isinstance(a, dict) else a["is_correct"]), False)
                if ws:
                    q_skipped += 1
                    bg = a["best_guess"] if isinstance(a, dict) else a["best_guess"]
                    gc = a["guess_correct"] if isinstance(a, dict) else a["guess_correct"]
                    if bg:
                        q_guessed += 1
                        if gc:
                            q_gc += 1
                elif a["is_correct"] if isinstance(a, dict) else a["is_correct"]:
                    q_correct += 1
                else:
                    q_wrong += 1
                q_marks += ma
                q_intuition += mi
                subj_m[subj] = round(subj_m.get(subj, 0.0) + ma, 2)

            quiz_history.append({
                "date": d_date,
                "marks": round(q_marks, 2),
                "marks_max": max_m,
                "score_pct": round(q_marks / max_m * 100, 1) if max_m else 0,
                "correct": q_correct, "wrong": q_wrong, "skipped": q_skipped,
                "guessed": q_guessed, "guess_correct": q_gc,
                "intuition_marks": round(q_intuition, 2),
                "delta_pct": round(q_intuition / max_m * 100, 1) if max_m else 0,
                "subjects": {s: {"marks": v} for s, v in subj_m.items()},
            })
    except Exception as e:
        print(f"[REPORT ERROR - quiz_history] {e}")

    # ---- suggested_reading: wrong topics from last 5 attempted sets ----
    suggested_reading = []
    try:
        last5_dates = [h["date"] for h in quiz_history[:5]]
        if last5_dates:
            ph5 = ",".join(["%s" if USE_PG else "?"] * len(last5_dates))
            if USE_PG:
                cur4 = conn.cursor()
                cur4.execute(f"""
                    SELECT json_array_elements_text(question_ids::json) AS q_id
                    FROM daily_sets WHERE date::text IN ({ph5})
                """, tuple(last5_dates))
                last5_qids = list(set(r[0] for r in cur4.fetchall()))
            else:
                cur = _execute(conn, f"""
                    SELECT value FROM daily_sets, json_each(question_ids)
                    WHERE date IN ({ph5})
                """, last5_dates)
                last5_qids = list(set(r[0] if USE_PG else r["value"] for r in _fetchall(cur, conn)))

            if last5_qids:
                phq = ",".join(["%s" if USE_PG else "?"] * len(last5_qids))
                if USE_PG:
                    cur5 = conn.cursor()
                    cur5.execute(f"""
                        SELECT DISTINCT ON (a.question_id)
                            q.upsc_topic, COALESCE(q.upsc_subject, q.subject, 'General') AS subject
                        FROM attempts a JOIN questions q ON a.question_id = q.id
                        WHERE a.question_id IN ({phq}) AND a.user_id = %s
                        AND a.is_correct = 0 AND (a.was_skipped IS NULL OR a.was_skipped = 0)
                        AND q.upsc_topic IS NOT NULL AND q.upsc_topic != ''
                        ORDER BY a.question_id, a.attempted_at DESC
                    """, tuple(last5_qids) + (uid,))
                    wrong_rows = [{"upsc_topic": r[0], "subject": r[1]} for r in cur5.fetchall()]
                else:
                    cur = _execute(conn, f"""
                        SELECT q.upsc_topic, COALESCE(q.upsc_subject, q.subject, 'General') AS subject
                        FROM attempts a JOIN questions q ON a.question_id = q.id
                        WHERE a.question_id IN ({phq}) AND a.user_id = ?
                        AND a.is_correct = 0 AND (a.was_skipped IS NULL OR a.was_skipped = 0)
                        AND q.upsc_topic IS NOT NULL AND q.upsc_topic != ''
                        GROUP BY a.question_id HAVING a.attempted_at = MAX(a.attempted_at)
                    """, last5_qids + [uid])
                    wrong_rows = _fetchall(cur, conn)

                topic_count: dict = {}
                for r in wrong_rows:
                    t = r["upsc_topic"] if isinstance(r, dict) else r["upsc_topic"]
                    s = _norm_subject((r["subject"] if isinstance(r, dict) else r["subject"]) or "")
                    key = (t, s)
                    topic_count[key] = topic_count.get(key, 0) + 1

                suggested_reading = [
                    {"topic": t, "subject": s, "wrong_count": c}
                    for (t, s), c in sorted(topic_count.items(), key=lambda x: -x[1])
                ][:8]
    except Exception as e:
        print(f"[REPORT ERROR - suggested_reading] {e}")

    conn.close()
    return {
        "total_attempts": total_attempts,
        "overall_accuracy_pct": None,
        "streak_days": streak,
        "subject_breakdown": subject_breakdown,
        "weak_areas": weak_areas,
        "marks": {
            "total_marks": round(total_marks_sum, 2),
            "max_marks": round(total_attempts * 2.0, 2),
            "score_pct": score_pct,
            "intuition_marks": round(total_intuition_sum, 2),
            "adjusted_pct": adjusted_pct,
            "delta_pct": delta_pct,
        },
        "quiz_history": quiz_history,
        "suggested_reading": suggested_reading,
    }


# ---------------------------------------------------------------------------
# Routes — Daily
# ---------------------------------------------------------------------------

@app.get("/daily")
def get_daily(date: Optional[str] = None, x_session_token: Optional[str] = Header(default=None)):
    target_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    is_today = (target_date == datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    conn = get_db()
    _check_and_auto_publish(conn)

    cur = _execute(conn, "SELECT question_ids FROM daily_sets WHERE date=?", (target_date,))
    row = _fetchone(cur, conn)
    if not row:
        conn.close()
        if is_today:
            raise HTTPException(503, detail={
                "status": "not_ready",
                "message": "Today's quiz will be available at 10 AM. Check the archive to practice past sets.",
            })
        raise HTTPException(404, f"No daily set found for {date}")

    q_ids = json.loads(row["question_ids"])
    user = _get_user_from_token(x_session_token, conn)

    # Fetch all questions in one query
    placeholders = ",".join(["?" if not USE_PG else "%s"] * len(q_ids))
    cur = _execute(conn, f"SELECT * FROM questions WHERE id IN ({placeholders})", tuple(q_ids))
    qrows = _fetchall(cur, conn)
    qmap = {r["id"]: r for r in qrows}
    questions = [_format_question(qmap[q_id]) for q_id in q_ids if q_id in qmap]

    # Fetch all attempts in one query.
    # NOTE: No is_daily filter — question_id IN (q_ids) already scopes to this set.
    # Filtering by is_daily=1 caused skip attempts (which sometimes saved with is_daily=0
    # due to UTC/IST boundary edge cases) to be silently excluded, breaking completion detection.
    attempts_map = {}
    if user:
        cur = _execute(conn,
            f"SELECT DISTINCT ON (question_id) question_id, chosen, is_correct, was_skipped, attempted_at "
            f"FROM attempts WHERE question_id IN ({placeholders}) AND user_id=? "
            f"ORDER BY question_id, attempted_at DESC"
            if USE_PG else
            f"SELECT question_id, chosen, is_correct, was_skipped, attempted_at FROM attempts "
            f"WHERE question_id IN ({placeholders}) AND user_id=? "
            f"GROUP BY question_id HAVING attempted_at = MAX(attempted_at)",
            tuple(q_ids) + (user["id"],))
        for a in _fetchall(cur, conn):
            attempts_map[a["question_id"]] = {
                "chosen": a["chosen"], "is_correct": bool(a["is_correct"]),
                "was_skipped": bool(a["was_skipped"]) if a["was_skipped"] is not None else False,
                "attempted_at": a["attempted_at"]
            }

    conn.close()
    attempted = len(attempts_map)
    correct = sum(1 for a in attempts_map.values() if a["is_correct"])
    return {
        "date": target_date, "questions": questions, "attempts": attempts_map,
        "summary": {"total": len(questions), "attempted": attempted,
                    "correct": correct, "completed": attempted == len(questions)},
    }


@app.get("/daily/archive")
def get_archive(detail: Optional[str] = None, x_session_token: Optional[str] = Header(default=None)):
    conn = get_db()
    user = _get_user_from_token(x_session_token, conn)
    signup_date = user["signup_date"] if user else None
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if detail:
        cur = _execute(conn, "SELECT question_ids FROM daily_sets WHERE date=?", (detail,))
        row = _fetchone(cur, conn)
        if not row:
            conn.close()
            raise HTTPException(404, f"No set found for {detail}")
        q_ids = json.loads(row["question_ids"])
        questions, attempts_map = [], {}
        for q_id in q_ids:
            cur = _execute(conn, "SELECT * FROM questions WHERE id=?", (q_id,))
            qrow = _fetchone(cur, conn)
            if qrow:
                questions.append(_format_question(qrow))
            if user:
                cur = _execute(conn,
                    "SELECT chosen, is_correct, attempted_at FROM attempts "
                    "WHERE question_id=? AND user_id=? ORDER BY attempted_at DESC LIMIT 1",
                    (q_id, user["id"]))
            else:
                cur = _execute(conn,
                    "SELECT chosen, is_correct, attempted_at FROM attempts "
                    "WHERE question_id=? ORDER BY attempted_at DESC LIMIT 1", (q_id,))
            a = _fetchone(cur, conn)
            if a:
                attempts_map[q_id] = {"chosen": a["chosen"], "is_correct": bool(a["is_correct"]),
                                       "attempted_at": a["attempted_at"]}
        conn.close()
        attempted_count = len(attempts_map)
        return {
            "date": detail, "state": "attempted" if attempted_count > 0 else "unattempted",
            "questions": questions, "attempts": attempts_map,
            "summary": {"total": len(questions), "attempted": attempted_count,
                        "correct": sum(1 for a in attempts_map.values() if a["is_correct"]),
                        "completed": attempted_count == len(questions)},
        }

    cur = _execute(conn, "SELECT date, question_ids FROM daily_sets ORDER BY date DESC")
    sets = _fetchall(cur, conn)

    result = []
    for s in sets:
        if s["date"] == today:
            continue
        q_ids = json.loads(s["question_ids"])
        placeholders = ",".join(["?"] * len(q_ids))
        user_and = "AND user_id = ?" if user else ""
        user_p = [user["id"]] if user else []

        cur = _execute(conn,
            f"SELECT COUNT(DISTINCT question_id) FROM attempts WHERE question_id IN ({placeholders}) {user_and}",
            q_ids + user_p)
        attempted = list(_fetchone(cur, conn).values())[0] if USE_PG else _fetchone(cur, conn)[0]

        cur = _execute(conn,
            f"SELECT COUNT(*) FROM attempts WHERE question_id IN ({placeholders}) AND is_correct=1 {user_and}",
            q_ids + user_p)
        correct = list(_fetchone(cur, conn).values())[0] if USE_PG else _fetchone(cur, conn)[0]

        result.append({
            "date": s["date"], "total": len(q_ids), "attempted": attempted,
            "correct": correct, "completed": attempted == len(q_ids),
            "state": "attempted" if attempted > 0 else "unattempted",
        })

    conn.close()
    return result


# ---------------------------------------------------------------------------
# Routes — Streak
# ---------------------------------------------------------------------------

@app.get("/streak/calendar")
def get_streak_calendar(x_session_token: Optional[str] = Header(default=None)):
    conn = get_db()
    user = _get_user_from_token(x_session_token, conn)
    today = datetime.now(timezone.utc).date()
    dates = [today + timedelta(days=i) for i in range(-3, 4)]

    completed_dates: set = set()   # all questions answered on publish date → streak tick
    partial_dates: set = set()     # some questions answered on publish date
    attempted_any_dates: set = set()  # any attempt exists (any date) → archive green

    if USE_PG:
        uid_param = (user["id"],) if user else None
        uid_check = "AND a.user_id = %s" if user else ""

        cur = conn.cursor()
        cur.execute(f"""
            SELECT ds.date FROM daily_sets ds
            WHERE (
                SELECT COUNT(DISTINCT a.question_id) FROM attempts a
                WHERE a.question_id = ANY(ARRAY(SELECT json_array_elements_text(ds.question_ids::json)))
                AND DATE(a.attempted_at::timestamp) = ds.date::date {uid_check}
            ) >= json_array_length(ds.question_ids::json)
        """, uid_param or ())
        completed_dates = {str(r[0]) for r in cur.fetchall()}

        cur.execute(f"""
            SELECT ds.date FROM daily_sets ds
            WHERE (
                SELECT COUNT(DISTINCT a.question_id) FROM attempts a
                WHERE a.question_id = ANY(ARRAY(SELECT json_array_elements_text(ds.question_ids::json)))
                AND DATE(a.attempted_at::timestamp) = ds.date::date {uid_check}
            ) > 0
        """, uid_param or ())
        partial_dates = {str(r[0]) for r in cur.fetchall()} - completed_dates

        if user:
            cur.execute("""
                SELECT ds.date FROM daily_sets ds
                WHERE (
                    SELECT COUNT(DISTINCT a.question_id) FROM attempts a
                    WHERE a.question_id = ANY(ARRAY(SELECT json_array_elements_text(ds.question_ids::json)))
                    AND a.user_id = %s
                ) > 0
            """, (user["id"],))
            attempted_any_dates = {str(r[0]) for r in cur.fetchall()}
    else:
        uid_check = "AND a.user_id = ?" if user else ""
        uid_p = [user["id"]] if user else []

        cur = _execute(conn, f"""
            SELECT ds.date FROM daily_sets ds
            WHERE (
                SELECT COUNT(DISTINCT a.question_id) FROM attempts a
                WHERE a.question_id IN (SELECT value FROM json_each(ds.question_ids))
                AND DATE(a.attempted_at) = ds.date {uid_check}
            ) >= json_array_length(ds.question_ids)
        """, uid_p)
        completed_dates = {r["date"] for r in _fetchall(cur, conn)}

        cur = _execute(conn, f"""
            SELECT ds.date FROM daily_sets ds
            WHERE (
                SELECT COUNT(DISTINCT a.question_id) FROM attempts a
                WHERE a.question_id IN (SELECT value FROM json_each(ds.question_ids))
                AND DATE(a.attempted_at) = ds.date {uid_check}
            ) > 0
        """, uid_p)
        partial_dates = {r["date"] for r in _fetchall(cur, conn)} - completed_dates

        if user:
            cur = _execute(conn, """
                SELECT ds.date FROM daily_sets ds
                WHERE (
                    SELECT COUNT(DISTINCT a.question_id) FROM attempts a
                    WHERE a.question_id IN (SELECT value FROM json_each(ds.question_ids))
                    AND a.user_id = ?
                ) > 0
            """, [user["id"]])
            attempted_any_dates = {r["date"] for r in _fetchall(cur, conn)}

    # Current streak: consecutive completed days going back from today (today counts if complete)
    streak = 0
    try:
        check = today if str(today) in completed_dates else today - timedelta(days=1)
        while str(check) in completed_dates:
            streak += 1
            check -= timedelta(days=1)
    except Exception as e:
        print(f"[STREAK ERROR] {e}")

    conn.close()
    return {
        "today": today.isoformat(),
        "streak": streak,
        "days": [
            {
                "date": d.isoformat(),
                "completed": d.isoformat() in completed_dates,
                "partial": d.isoformat() in partial_dates,
                # attempted on a different day than publish date (archive practice)
                "attempted_archive": (
                    d.isoformat() in attempted_any_dates
                    and d.isoformat() not in completed_dates
                    and d.isoformat() not in partial_dates
                ),
                "is_today": d == today,
                "is_future": d > today,
            }
            for d in dates
        ],
    }


# ---------------------------------------------------------------------------
# Routes — Misc
# ---------------------------------------------------------------------------

@app.get("/subjects")
def list_subjects():
    conn = get_db()
    cur = _execute(conn, "SELECT subject, COUNT(*) as count FROM questions GROUP BY subject ORDER BY count DESC")
    rows = _fetchall(cur, conn)
    conn.close()
    return [{"subject": r["subject"], "count": r["count"]} for r in rows]


@app.post("/daily/restart")
def restart_daily(date: Optional[str] = None, x_session_token: Optional[str] = Header(default=None)):
    target_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = get_db()
    user = _get_user_from_token(x_session_token, conn)
    cur = _execute(conn, "SELECT question_ids FROM daily_sets WHERE date=?", (target_date,))
    row = _fetchone(cur, conn)
    if not row:
        conn.close()
        raise HTTPException(404, "No daily set found for this date")
    q_ids = json.loads(row["question_ids"])
    placeholders = ",".join(["?"] * len(q_ids))
    if user:
        cur = _execute(conn, f"DELETE FROM attempts WHERE question_id IN ({placeholders}) AND user_id=?",
                       tuple(q_ids) + (user["id"],))
    else:
        cur = _execute(conn, f"DELETE FROM attempts WHERE question_id IN ({placeholders})", q_ids)
    deleted = cur.rowcount
    _commit(conn)
    conn.close()
    return {"date": target_date, "attempts_cleared": deleted}


@app.get("/stats")
def stats():
    conn = get_db()
    cur = _execute(conn, "SELECT COUNT(*) FROM questions"); total_q = list(_fetchone(cur,conn).values())[0] if USE_PG else _fetchone(cur,conn)[0]
    cur = _execute(conn, "SELECT COUNT(*) FROM questions WHERE status='pass'"); passed_q = list(_fetchone(cur,conn).values())[0] if USE_PG else _fetchone(cur,conn)[0]
    cur = _execute(conn, "SELECT COUNT(*) FROM attempts"); total_a = list(_fetchone(cur,conn).values())[0] if USE_PG else _fetchone(cur,conn)[0]
    conn.close()
    return {"total_questions": total_q, "passed_questions": passed_q, "total_attempts": total_a}


# ---------------------------------------------------------------------------
# Internal / admin routes
# ---------------------------------------------------------------------------

@app.get("/internal/debug/user-state")
def admin_debug_user_state(email: str, secret: str = ""):
    """Admin diagnostic: full quiz state for a user by email. Requires PIPELINE_SECRET."""
    if PIPELINE_SECRET and secret != PIPELINE_SECRET:
        raise HTTPException(403, "Forbidden")
    conn = get_db()
    cur = _execute(conn, "SELECT id FROM users WHERE email=?", (email.lower(),))
    user_row = _fetchone(cur, conn)
    if not user_row:
        conn.close()
        return {"error": f"no user found for {email}"}
    uid = user_row["id"] if isinstance(user_row, dict) else user_row[0]

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cur = _execute(conn, "SELECT question_ids FROM daily_sets WHERE date=?", (today,))
    ds_row = _fetchone(cur, conn)
    if not ds_row:
        conn.close()
        return {"error": "no daily set for today", "today": today}
    q_ids = json.loads(ds_row["question_ids"]) if isinstance(ds_row["question_ids"], str) else ds_row["question_ids"]

    ph = ",".join(["%s" if USE_PG else "?"] * len(q_ids))
    if USE_PG:
        cur = conn.cursor()
        cur.execute(
            f"SELECT question_id, chosen, is_correct, was_skipped, is_daily, best_guess, "
            f"marks_actual, marks_intuition, attempted_at "
            f"FROM attempts WHERE question_id IN ({ph}) AND user_id = %s ORDER BY attempted_at DESC",
            tuple(q_ids) + (uid,)
        )
        cols = [d[0] for d in cur.description]
        daily_attempts = [dict(zip(cols, r)) for r in cur.fetchall()]

        cur.execute(
            "SELECT question_id, was_skipped, best_guess, guess_correct, marks_intuition, is_daily, attempted_at "
            "FROM attempts WHERE user_id = %s AND was_skipped = 1 ORDER BY attempted_at DESC LIMIT 20",
            (uid,)
        )
        cols = [d[0] for d in cur.description]
        recent_skips = [dict(zip(cols, r)) for r in cur.fetchall()]
    else:
        cur = _execute(conn,
            f"SELECT question_id, chosen, is_correct, was_skipped, is_daily, best_guess, "
            f"marks_actual, marks_intuition, attempted_at "
            f"FROM attempts WHERE question_id IN ({ph}) AND user_id=? ORDER BY attempted_at DESC",
            tuple(q_ids) + (uid,))
        daily_attempts = [dict(r) for r in _fetchall(cur, conn)]
        cur = _execute(conn,
            "SELECT question_id, was_skipped, best_guess, guess_correct, marks_intuition, is_daily, attempted_at "
            "FROM attempts WHERE user_id=? AND was_skipped=1 ORDER BY attempted_at DESC LIMIT 20",
            (uid,))
        recent_skips = [dict(r) for r in _fetchall(cur, conn)]

    conn.close()
    answered = {a["question_id"] for a in daily_attempts}
    return {
        "user_id": uid, "email": email, "today": today,
        "daily_set_size": len(q_ids),
        "daily_attempts_count": len(answered),
        "completed": len(answered) == len(q_ids),
        "daily_attempts": daily_attempts,
        "recent_skips_last20": recent_skips,
    }


@app.post("/internal/stage-batch")
def stage_batch(body: StageBatchIn):
    if PIPELINE_SECRET and body.secret != PIPELINE_SECRET:
        raise HTTPException(403, "Forbidden")
    conn = get_db()
    staged_at = datetime.now(timezone.utc)
    auto_publish_at = staged_at + timedelta(hours=1)
    if USE_PG:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO review_batches (date, question_ids, staged_at, auto_publish_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (date) DO UPDATE SET
                question_ids=EXCLUDED.question_ids,
                staged_at=EXCLUDED.staged_at,
                auto_publish_at=EXCLUDED.auto_publish_at,
                published_at=NULL, published_by=NULL
        """, (body.date, json.dumps(body.question_ids), staged_at.isoformat(), auto_publish_at.isoformat()))
    else:
        conn.execute(
            "INSERT OR REPLACE INTO review_batches (date, question_ids, staged_at, auto_publish_at) VALUES (?,?,?,?)",
            (body.date, json.dumps(body.question_ids), staged_at.isoformat(), auto_publish_at.isoformat()))
    ph = ",".join(["%s" if USE_PG else "?"] * len(body.question_ids))
    _execute(conn, f"UPDATE questions SET review_decision='accept' WHERE id IN ({ph})", tuple(body.question_ids))
    _commit(conn)
    conn.close()
    return {"date": body.date, "questions": len(body.question_ids),
            "staged_at": staged_at.isoformat(), "auto_publish_at": auto_publish_at.isoformat()}


@app.get("/internal/review/pending")
def get_pending_review(secret: str = ""):
    if PIPELINE_SECRET and secret != PIPELINE_SECRET:
        raise HTTPException(403, "Forbidden")
    conn = get_db()
    _check_and_auto_publish(conn)
    cur = _execute(conn,
        "SELECT date, question_ids, staged_at, auto_publish_at, published_at, published_by, prompt_notes "
        "FROM review_batches ORDER BY date DESC LIMIT 1")
    batch = _fetchone(cur, conn)
    if not batch:
        conn.close()
        return {"batch": None, "questions": []}
    q_ids = json.loads(batch["question_ids"]) if isinstance(batch["question_ids"], str) else batch["question_ids"]
    ph = ",".join(["%s" if USE_PG else "?"] * len(q_ids))
    cur = _execute(conn, f"SELECT * FROM questions WHERE id IN ({ph})", tuple(q_ids))
    qrows = _fetchall(cur, conn)
    qmap = {r["id"]: r for r in qrows}
    questions = []
    for q_id in q_ids:
        if q_id not in qmap:
            continue
        q = qmap[q_id]
        fmt = _format_question(q)
        fmt["review_decision"] = q.get("review_decision") or "accept"
        fmt["flag_reason"] = q.get("flag_reason")
        fmt["status"] = q.get("status")
        subject_val = (q.get("upsc_subject") or "").strip() or (q.get("subject") or "").strip()
        if not subject_val:
            subject_val = _infer_subject(dict(q))
            if subject_val == "General Studies":
                subject_val = ""
        fmt["subject"] = subject_val
        questions.append(fmt)
    conn.close()
    return {
        "batch": {
            "date": batch["date"],
            "staged_at": batch["staged_at"],
            "auto_publish_at": batch["auto_publish_at"],
            "published_at": batch["published_at"],
            "published_by": batch["published_by"],
            "prompt_notes": batch["prompt_notes"],
        },
        "questions": questions,
    }


@app.patch("/internal/review/question/{question_id}")
def update_review_question(question_id: str, body: ReviewQuestionUpdateIn):
    if PIPELINE_SECRET and body.secret != PIPELINE_SECRET:
        raise HTTPException(403, "Forbidden")
    conn = get_db()
    updates, params = [], []
    if body.decision is not None:
        updates.append("review_decision = ?"); params.append(body.decision)
    if body.upsc_subject is not None:
        updates.append("upsc_subject = ?"); params.append(body.upsc_subject)
    if body.question is not None:
        updates.append("question = ?"); params.append(body.question)
    if body.options is not None:
        updates.append("options = ?"); params.append(json.dumps(body.options))
    if body.explanation is not None:
        updates.append("explanation = ?"); params.append(body.explanation)
    if updates:
        params.append(question_id)
        _execute(conn, f"UPDATE questions SET {', '.join(updates)} WHERE id = ?", tuple(params))
        _commit(conn)
    conn.close()
    return {"ok": True}


@app.post("/internal/review/publish")
def publish_review(body: ReviewPublishIn):
    if PIPELINE_SECRET and body.secret != PIPELINE_SECRET:
        raise HTTPException(403, "Forbidden")
    conn = get_db()
    cur = _execute(conn, "SELECT question_ids FROM review_batches WHERE date=?", (body.date,))
    batch = _fetchone(cur, conn)
    if not batch:
        conn.close()
        raise HTTPException(404, "No batch found for this date")
    q_ids = json.loads(batch["question_ids"]) if isinstance(batch["question_ids"], str) else batch["question_ids"]
    ph = ",".join(["%s" if USE_PG else "?"] * len(q_ids))
    cur = _execute(conn,
        f"SELECT id FROM questions WHERE id IN ({ph}) AND COALESCE(review_decision,'accept') != 'reject'",
        tuple(q_ids))
    accepted = [r["id"] for r in _fetchall(cur, conn)]
    selected = _topup_from_pool(conn, accepted[:10], body.date)
    now_iso = datetime.now(timezone.utc).isoformat()
    _upsert_daily_set(conn, body.date, json.dumps(selected), now_iso)
    _execute(conn,
        "UPDATE review_batches SET published_at=?, published_by='human', prompt_notes=? WHERE date=?",
        (now_iso, body.prompt_notes, body.date))
    _commit(conn)
    conn.close()
    return {"date": body.date, "selected": len(selected), "accepted": len(accepted),
            "auto_filled": max(0, len(selected) - min(len(accepted), 10)), "published_by": "human"}


@app.get("/internal/review/last-notes")
def get_last_review_notes(secret: str = ""):
    """Return prompt_notes from the most recently published review batch."""
    if PIPELINE_SECRET and secret != PIPELINE_SECRET:
        raise HTTPException(403, "Forbidden")
    conn = get_db()
    cur = _execute(conn,
        "SELECT prompt_notes FROM review_batches WHERE published_at IS NOT NULL "
        "ORDER BY published_at DESC LIMIT 1")
    row = _fetchone(cur, conn)
    conn.close()
    notes = row["prompt_notes"] if row else None
    return {"prompt_notes": notes or ""}


@app.get("/internal/otp/{email}")
def get_otp(email: str):
    conn = get_db()
    cur = _execute(conn, "SELECT code, expires_at FROM otps WHERE email=?", (email.lower(),))
    row = _fetchone(cur, conn)
    conn.close()
    if not row:
        raise HTTPException(404, "No OTP found for this email")
    return {"email": email, "code": row["code"], "expires_at": row["expires_at"]}


@app.get("/internal/users")
def list_users():
    conn = get_db()
    cur = _execute(conn, "SELECT email, signup_date, created_at FROM users ORDER BY created_at DESC")
    rows = _fetchall(cur, conn)
    cur = _execute(conn, "SELECT COUNT(*) FROM attempts WHERE user_id IS NOT NULL")
    total_a = list(_fetchone(cur,conn).values())[0] if USE_PG else _fetchone(cur,conn)[0]
    conn.close()
    return {
        "total_users": len(rows), "total_attempts": total_a,
        "users": [{"email": r["email"], "signup_date": r["signup_date"], "created_at": r["created_at"]} for r in rows],
    }


@app.post("/internal/sync")
def internal_sync():
    inserted, skipped = import_questions()
    updated = sync_statuses()
    return {"inserted": inserted, "skipped": skipped, "statuses_synced": updated}


@app.post("/internal/backfill-missing-skips")
def backfill_missing_skips(secret: str = ""):
    """One-time backfill: insert skip records for questions with no attempt in each daily set.
    Only runs on past sets (not today). Skips users who never touched that set."""
    if PIPELINE_SECRET and secret != PIPELINE_SECRET:
        raise HTTPException(403, "Forbidden")

    conn = get_db()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cur = _execute(conn, "SELECT date, question_ids FROM daily_sets WHERE date < ? ORDER BY date", (today,))
    sets = _fetchall(cur, conn)

    total_inserted = 0
    now_iso = datetime.now(timezone.utc).isoformat()

    for ds in sets:
        date = ds["date"]
        q_ids = json.loads(ds["question_ids"]) if isinstance(ds["question_ids"], str) else ds["question_ids"]
        if not q_ids:
            continue

        ph = ",".join(["%s" if USE_PG else "?"] * len(q_ids))

        # Users who have at least one attempt on this set
        cur = _execute(conn,
            f"SELECT DISTINCT user_id FROM attempts WHERE question_id IN ({ph}) AND user_id IS NOT NULL",
            tuple(q_ids))
        users = [r["user_id"] for r in _fetchall(cur, conn)]

        for uid in users:
            cur = _execute(conn,
                f"SELECT DISTINCT question_id FROM attempts WHERE question_id IN ({ph}) AND user_id = {'%s' if USE_PG else '?'}",
                tuple(q_ids) + (uid,))
            attempted_ids = {r["question_id"] for r in _fetchall(cur, conn)}

            missing = [q_id for q_id in q_ids if q_id not in attempted_ids]
            for q_id in missing:
                _execute(conn,
                    "INSERT INTO attempts (id, question_id, chosen, is_correct, time_taken, attempted_at, "
                    "user_id, is_daily, was_skipped, best_guess, guess_correct, marks_actual, marks_intuition) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (str(uuid.uuid4()), q_id, "", 0, 0, now_iso,
                     uid, 1, 1, None, None, 0.0, 0.0))
                total_inserted += 1

    _commit(conn)
    conn.close()
    return {"sets_processed": len(sets), "skips_inserted": total_inserted}


@app.post("/internal/push-daily-set")
def push_daily_set(body: PushDailySetIn):
    if PIPELINE_SECRET and body.secret != PIPELINE_SECRET:
        raise HTTPException(403, "Forbidden")
    conn = get_db()
    _upsert_daily_set(conn, body.date, json.dumps(body.question_ids), datetime.now(timezone.utc).isoformat())
    _commit(conn)
    conn.close()
    return {"date": body.date, "questions": len(body.question_ids)}


@app.post("/internal/push-questions")
def push_questions(body: PushQuestionsIn):
    if PIPELINE_SECRET and body.secret != PIPELINE_SECRET:
        raise HTTPException(403, "Forbidden")
    conn = get_db()
    inserted = skipped = 0
    for q in body.questions:
        q_id = q.get("id") or str(uuid.uuid5(uuid.NAMESPACE_DNS, q.get("question", "") + q.get("source_file", "")))
        cur = _execute(conn, "SELECT id FROM questions WHERE id=?", (q_id,))
        if _fetchone(cur, conn):
            if body.update_status and q.get("status"):
                _execute(conn, "UPDATE questions SET status=?, flag_reason=? WHERE id=?",
                         (q.get("status"), q.get("flag_reason"), q_id))
                skipped += 1
            else:
                skipped += 1
            continue
        extracts = q.get("cited_extracts") or ([q["cited_extract"]] if q.get("cited_extract") else [])
        _execute(conn, """
            INSERT INTO questions
              (id, question, options, correct, explanation, subject, difficulty,
               question_type, source_type, source_file, source_page,
               status, flag_reason, extracts, raw,
               upsc_subject, upsc_topic, broad_category, question_category, suggested_reading)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            q_id, q.get("question", ""), json.dumps(q.get("options", {})),
            q.get("correct_answer") or q.get("correct", ""), q.get("explanation", ""),
            q.get("subject", ""), q.get("difficulty", "medium"),
            q.get("question_type", "statement_based"),
            q.get("source_type", "ncert"), q.get("source_file", ""),
            _safe_int(q.get("source_page")), q.get("status", "unchecked"),
            q.get("flag_reason"), json.dumps(extracts), json.dumps(q),
            q.get("upsc_subject"), q.get("upsc_topic"), q.get("broad_category"),
            q.get("question_category"), json.dumps(q.get("suggested_reading")) if q.get("suggested_reading") else None,
        ))
        inserted += 1
    _commit(conn)
    conn.close()
    return {"inserted": inserted, "skipped": skipped}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_question(row) -> dict:
    sr_raw = row.get("suggested_reading") if isinstance(row, dict) else (row["suggested_reading"] if "suggested_reading" in row.keys() else None)
    suggested_reading = json.loads(sr_raw) if sr_raw else None
    return {
        "id": row["id"], "question": row["question"],
        "options": json.loads(row["options"]) if isinstance(row["options"], str) else row["options"],
        "correct_answer": row["correct"], "explanation": row["explanation"],
        "difficulty": row["difficulty"], "question_type": row["question_type"],
        "source_type": row["source_type"], "upsc_subject": row["upsc_subject"],
        "upsc_topic": row["upsc_topic"], "upsc_category": row["broad_category"],
        "question_category": row["question_category"],
        "suggested_reading": suggested_reading,
        "is_pyq": "pyq" in (row["source_file"] or "").lower(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=PORT, reload=True)
