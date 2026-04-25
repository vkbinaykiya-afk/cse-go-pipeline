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
STATIC_SUBJECTS   = ["History", "Geography", "Polity", "Environment",
                     "Science & Technology", "Economics", "Current Affairs"]

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
                id           TEXT PRIMARY KEY,
                question_id  TEXT NOT NULL,
                chosen       TEXT NOT NULL,
                is_correct   INTEGER NOT NULL,
                time_taken   INTEGER,
                attempted_at TEXT NOT NULL,
                user_id      TEXT,
                is_daily     INTEGER NOT NULL DEFAULT 0
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
                id TEXT PRIMARY KEY, question_id TEXT NOT NULL, chosen TEXT NOT NULL,
                is_correct INTEGER NOT NULL, time_taken INTEGER, attempted_at TEXT NOT NULL,
                user_id TEXT, is_daily INTEGER NOT NULL DEFAULT 0
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
        """)
        conn.commit()
    conn.close()


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
    chosen: str
    time_taken: Optional[int] = None

class AttemptOut(BaseModel):
    id: str
    question_id: str
    chosen: str
    is_correct: bool
    correct_answer: str
    explanation: str

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

@app.on_event("startup")
def startup():
    init_db()
    inserted, skipped = import_questions()
    updated = sync_statuses()
    print(f"DB ready ({'postgres' if USE_PG else 'sqlite'}). "
          f"Imported {inserted} new, {skipped} existing, {updated} statuses synced.")


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
    conn = get_db()
    user = _get_user_from_token(x_session_token, conn)
    cur = _execute(conn, "SELECT * FROM questions WHERE id=?", (body.question_id,))
    q = _fetchone(cur, conn)
    if not q:
        conn.close()
        raise HTTPException(404, "Question not found")

    is_correct = body.chosen.upper() == q["correct"].upper()
    attempt_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cur = _execute(conn, "SELECT question_ids FROM daily_sets WHERE date=?", (today,))
    daily_row = _fetchone(cur, conn)
    today_ids = set(json.loads(daily_row["question_ids"])) if daily_row else set()
    is_daily = 1 if body.question_id in today_ids else 0

    _execute(conn,
        "INSERT INTO attempts (id, question_id, chosen, is_correct, time_taken, attempted_at, user_id, is_daily) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (attempt_id, body.question_id, body.chosen, int(is_correct), body.time_taken,
         now, user["id"] if user else None, is_daily))
    _commit(conn)
    conn.close()

    return AttemptOut(
        id=attempt_id, question_id=body.question_id, chosen=body.chosen,
        is_correct=is_correct, correct_answer=q["correct"], explanation=q["explanation"] or "",
    )


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
    cur = _execute(conn, "SELECT COUNT(*) FROM attempts WHERE user_id = ?", (uid,))
    total_attempts = _fetchone(cur, conn)
    total_attempts = list(total_attempts.values())[0] if USE_PG else total_attempts[0]

    cur = _execute(conn, "SELECT COUNT(*) FROM attempts WHERE user_id = ? AND is_correct=1", (uid,))
    correct_attempts = _fetchone(cur, conn)
    correct_attempts = list(correct_attempts.values())[0] if USE_PG else correct_attempts[0]

    cur = _execute(conn, """
        SELECT COALESCE(q.upsc_subject, q.subject, 'Unknown') AS subject,
               COUNT(a.id) AS total, SUM(a.is_correct) AS correct,
               ROUND(AVG(a.is_correct::float)*100, 1) AS accuracy_pct
        FROM attempts a JOIN questions q ON a.question_id = q.id
        WHERE a.user_id = ? GROUP BY subject
    """ if USE_PG else """
        SELECT COALESCE(q.upsc_subject, q.subject, 'Unknown') AS subject,
               COUNT(a.id) AS total, SUM(a.is_correct) AS correct,
               ROUND(AVG(a.is_correct)*100, 1) AS accuracy_pct
        FROM attempts a JOIN questions q ON a.question_id = q.id
        WHERE a.user_id = ? GROUP BY subject
    """, (uid,))
    attempted_map = {r["subject"]: r for r in _fetchall(cur, conn)}

    subject_breakdown = []
    for subj in STATIC_SUBJECTS:
        if subj in attempted_map:
            r = attempted_map[subj]
            subject_breakdown.append({
                "subject": subj, "total": r["total"], "correct": r["correct"],
                "accuracy_pct": float(r["accuracy_pct"]) if r["accuracy_pct"] else 0,
                "status": "attempted",
            })
        else:
            subject_breakdown.append({
                "subject": subj, "total": 0, "correct": 0,
                "accuracy_pct": None, "status": "not_attempted",
            })

    weak_areas = sorted(
        [s for s in subject_breakdown if s["status"] == "attempted" and s["total"] >= 2],
        key=lambda s: s["accuracy_pct"]
    )[:3]

    # Streak — completed days (all questions answered on that day)
    if USE_PG:
        cur = conn.cursor()
        cur.execute("""
            SELECT ds.date FROM daily_sets ds
            WHERE (
                SELECT COUNT(DISTINCT a.question_id) FROM attempts a
                WHERE a.question_id = ANY(ARRAY(SELECT json_array_elements_text(ds.question_ids::json)))
                AND DATE(a.attempted_at::timestamp) = ds.date::date
                AND a.user_id = %s
            ) >= json_array_length(ds.question_ids::json)
            ORDER BY ds.date DESC
        """, (uid,))
        completed_days = [{"date": r[0]} for r in cur.fetchall()]
    else:
        cur = _execute(conn, """
            SELECT ds.date FROM daily_sets ds
            WHERE (
                SELECT COUNT(DISTINCT a.question_id) FROM attempts a
                WHERE a.question_id IN (SELECT value FROM json_each(ds.question_ids))
                AND DATE(a.attempted_at) = ds.date AND a.user_id = ?
            ) >= json_array_length(ds.question_ids)
            ORDER BY ds.date DESC
        """, (uid,))
        completed_days = _fetchall(cur, conn)

    streak = 0
    prev = None
    for row in completed_days:
        d = str(row["date"])
        if prev is None or (datetime.fromisoformat(prev) - datetime.fromisoformat(d)).days == 1:
            streak += 1; prev = d
        else:
            break

    conn.close()
    return {
        "total_attempts": total_attempts,
        "overall_accuracy_pct": round(correct_attempts / total_attempts * 100, 1) if total_attempts else 0,
        "streak_days": streak,
        "subject_breakdown": subject_breakdown,
        "weak_areas": weak_areas,
    }


# ---------------------------------------------------------------------------
# Routes — Daily
# ---------------------------------------------------------------------------

@app.get("/daily")
def get_daily(date: Optional[str] = None, x_session_token: Optional[str] = Header(default=None)):
    target_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    is_today = (target_date == datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    conn = get_db()

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

    questions = []
    for q_id in q_ids:
        cur = _execute(conn, "SELECT * FROM questions WHERE id=?", (q_id,))
        qrow = _fetchone(cur, conn)
        if qrow:
            questions.append(_format_question(qrow))

    attempts_map = {}
    for q_id in q_ids:
        if user:
            cur = _execute(conn,
                "SELECT chosen, is_correct, attempted_at FROM attempts "
                "WHERE question_id=? AND is_daily=1 AND user_id=? ORDER BY attempted_at DESC LIMIT 1",
                (q_id, user["id"]))
        else:
            cur = _execute(conn,
                "SELECT chosen, is_correct, attempted_at FROM attempts "
                "WHERE question_id=? AND is_daily=1 ORDER BY attempted_at DESC LIMIT 1",
                (q_id,))
        a = _fetchone(cur, conn)
        if a:
            attempts_map[q_id] = {"chosen": a["chosen"], "is_correct": bool(a["is_correct"]),
                                   "attempted_at": a["attempted_at"]}

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

    if signup_date:
        cur = _execute(conn, "SELECT date, question_ids FROM daily_sets WHERE date >= ? ORDER BY date DESC", (signup_date,))
    else:
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

    conn.close()
    return {
        "today": today.isoformat(),
        "days": [
            {"date": d.isoformat(), "completed": d.isoformat() in completed_dates,
             "partial": d.isoformat() in partial_dates,
             "is_today": d == today, "is_future": d > today}
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
def restart_daily(date: Optional[str] = None):
    target_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = get_db()
    cur = _execute(conn, "SELECT question_ids FROM daily_sets WHERE date=?", (target_date,))
    row = _fetchone(cur, conn)
    if not row:
        conn.close()
        raise HTTPException(404, "No daily set found for this date")
    q_ids = json.loads(row["question_ids"])
    placeholders = ",".join(["?"] * len(q_ids))
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
