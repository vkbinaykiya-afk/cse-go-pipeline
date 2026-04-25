"""
FastAPI backend for CSE-GO quiz app.
Serves questions, records attempts, returns personalized reports.
"""

import json
import sqlite3
import glob
import os
import uuid
import secrets
import string
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from pathlib import Path

import smtplib
import urllib.request
from email.mime.text import MIMEText
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

GMAIL_USER = os.environ.get("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
RESEND_FROM = os.environ.get("RESEND_FROM", "onboarding@resend.dev")
OTP_EXPIRY_MINUTES = 10
PORT = int(os.environ.get("PORT", 8000))
PIPELINE_SECRET = os.environ.get("PIPELINE_SECRET", "")
STATIC_SUBJECTS = ["History", "Geography", "Polity", "Environment",
                   "Science & Technology", "Economics", "Current Affairs"]

DB_PATH = Path(__file__).parent / "cse_go.db"
QUESTIONS_DIR = Path(__file__).parent / "questions"

app = FastAPI(title="CSE-GO API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS questions (
            id              TEXT PRIMARY KEY,
            question        TEXT NOT NULL,
            options         TEXT NOT NULL,      -- JSON object
            correct         TEXT NOT NULL,
            explanation     TEXT,
            subject         TEXT,               -- legacy heuristic subject
            difficulty      TEXT,
            question_type   TEXT,
            source_type     TEXT,
            source_file     TEXT,
            source_page     INTEGER,
            status          TEXT,
            flag_reason     TEXT,
            extracts        TEXT,               -- JSON array
            raw             TEXT NOT NULL,      -- full original JSON
            -- UPSC tagging (populated by tag_questions.py)
            upsc_subject    TEXT,               -- Polity | History | Geography | Economy | Environment | Science & Tech | Current Affairs
            upsc_topic      TEXT,               -- UPSC syllabus topic, e.g. "Fundamental Rights"
            broad_category  TEXT,               -- parent grouping, e.g. "Part III of Constitution"
            question_category TEXT              -- factual | conceptual | trend_based | in_news | map_based
        );

        CREATE TABLE IF NOT EXISTS attempts (
            id          TEXT PRIMARY KEY,
            question_id TEXT NOT NULL,
            chosen      TEXT NOT NULL,
            is_correct  INTEGER NOT NULL,
            time_taken  INTEGER,
            attempted_at TEXT NOT NULL,
            FOREIGN KEY (question_id) REFERENCES questions(id)
        );

        CREATE TABLE IF NOT EXISTS daily_sets (
            date         TEXT PRIMARY KEY,   -- YYYY-MM-DD
            question_ids TEXT NOT NULL,      -- JSON array of question IDs
            created_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS users (
            id           TEXT PRIMARY KEY,
            email        TEXT UNIQUE NOT NULL,
            signup_date  TEXT NOT NULL,      -- YYYY-MM-DD
            created_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS otps (
            email       TEXT PRIMARY KEY,
            code        TEXT NOT NULL,
            expires_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            token      TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    # Migrate: add columns if they don't exist yet
    for col, typedef in [
        ("upsc_subject",      "TEXT"),
        ("upsc_topic",        "TEXT"),
        ("broad_category",    "TEXT"),
        ("question_category", "TEXT"),
        ("generated_at",      "TEXT"),
        ("checked_at",        "TEXT"),
        ("repaired_at",       "TEXT"),
        ("pipeline_version",  "TEXT"),
        ("topic_query",       "TEXT"),
        ("suggested_reading", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE questions ADD COLUMN {col} {typedef}")
        except sqlite3.OperationalError:
            pass
    # Add user_id to attempts if missing
    try:
        conn.execute("ALTER TABLE attempts ADD COLUMN user_id TEXT")
    except sqlite3.OperationalError:
        pass
    # is_daily: 1 if this attempt was part of today's scheduled daily set
    try:
        conn.execute("ALTER TABLE attempts ADD COLUMN is_daily INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()


def sync_statuses():
    """Sync status/flag_reason for existing questions from JSON files into DB."""
    conn = get_db()
    files = glob.glob(str(QUESTIONS_DIR / "*.json"))
    updated = 0
    for fpath in files:
        with open(fpath) as f:
            try:
                data = json.load(f)
            except Exception:
                continue
        if isinstance(data, dict):
            data = [data]
        for q in data:
            q_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, q.get("question", "") + q.get("source_file", "")))
            status = q.get("status")
            if status in ("pass", "flag"):
                conn.execute(
                    "UPDATE questions SET status=?, flag_reason=? WHERE id=?",
                    (status, q.get("flag_reason"), q_id)
                )
                updated += 1
    conn.commit()
    conn.close()
    return updated


def import_questions():
    """Import all question JSON files from ./questions/ into DB."""
    conn = get_db()
    files = glob.glob(str(QUESTIONS_DIR / "*.json"))
    inserted = 0
    skipped = 0
    for fpath in files:
        with open(fpath) as f:
            try:
                data = json.load(f)
            except Exception:
                continue
        if isinstance(data, dict):
            data = [data]
        for q in data:
            # Derive a stable ID from question text
            q_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, q.get("question", "") + q.get("source_file", "")))
            existing = conn.execute("SELECT id FROM questions WHERE id=?", (q_id,)).fetchone()
            if existing:
                skipped += 1
                continue

            # Derive subject from source_file heuristic
            subject = _infer_subject(q)

            extracts = q.get("cited_extracts") or ([q["cited_extract"]] if q.get("cited_extract") else [])

            conn.execute("""
                INSERT INTO questions
                  (id, question, options, correct, explanation, subject, difficulty,
                   question_type, source_type, source_file, source_page,
                   status, flag_reason, extracts, raw)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                q_id,
                q.get("question", ""),
                json.dumps(q.get("options", {})),
                q.get("correct_answer", ""),
                q.get("explanation", ""),
                subject,
                q.get("difficulty", "medium"),
                q.get("question_type", "statement_based"),
                q.get("source_type", "ncert"),
                q.get("source_file", ""),
                _safe_int(q.get("source_page")),
                q.get("status", "unchecked"),
                q.get("flag_reason"),
                json.dumps(extracts),
                json.dumps(q),
            ))
            inserted += 1

    conn.commit()
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
    source_type = q.get("source_type", "")
    mapping = {
        "polity": "Polity",
        "political": "Polity",
        "constitution": "Polity",
        "geography": "Geography",
        "contemporary_india": "Geography",
        "history": "History",
        "economics": "Economics",
        "economy": "Economics",
        "environment": "Environment",
        "science": "Science & Technology",
        "biology": "Science & Technology",
        "physics": "Science & Technology",
        "chemistry": "Science & Technology",
        "current_affairs": "Current Affairs",
        "finance commission": "Economy",
        "ramsar": "Environment",
    }
    for key, subject in mapping.items():
        if key in src:
            return subject
    if source_type == "ca":
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


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _get_user_from_token(token: Optional[str], conn) -> Optional[dict]:
    if not token:
        return None
    row = conn.execute(
        "SELECT u.id, u.email, u.signup_date FROM sessions s "
        "JOIN users u ON s.user_id = u.id WHERE s.token = ?", (token,)
    ).fetchone()
    return dict(row) if row else None


def _send_otp_email(email: str, code: str):
    body = f"Your CSE-GO one-time login code is: {code}\n\nValid for {OTP_EXPIRY_MINUTES} minutes."

    # Resend HTTP API — works on all cloud hosts (no SMTP port restrictions)
    if RESEND_API_KEY:
        import json as _json
        payload = _json.dumps({
            "from": RESEND_FROM,
            "to": [email],
            "subject": "Your CSE-GO login code",
            "text": body,
        }).encode()
        req = urllib.request.Request(
            "https://api.resend.com/emails",
            data=payload,
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Resend API error {resp.status}")
        return

    # Gmail SMTP — Mac local dev only
    msg = MIMEText(body)
    msg["Subject"] = "Your CSE-GO login code"
    msg["From"] = GMAIL_USER
    msg["To"] = email
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as s:
        s.starttls()
        s.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        s.send_message(msg)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup():
    init_db()
    inserted, skipped = import_questions()
    updated = sync_statuses()
    print(f"DB ready. Imported {inserted} new, {skipped} existing, {updated} statuses synced.")


@app.get("/questions")
def list_questions(
    subject: Optional[str] = None,
    upsc_subject: Optional[str] = None,
    upsc_topic: Optional[str] = None,
    question_category: Optional[str] = None,
    difficulty: Optional[str] = None,
    question_type: Optional[str] = None,
    status: Optional[str] = Query(default="pass"),   # pass | flag | unchecked | all
    limit: int = 10,
    offset: int = 0,
):
    conn = get_db()
    clauses = []
    params = []

    if subject:
        clauses.append("subject = ?")
        params.append(subject)
    if upsc_subject:
        clauses.append("upsc_subject = ?")
        params.append(upsc_subject)
    if upsc_topic:
        clauses.append("upsc_topic = ?")
        params.append(upsc_topic)
    if question_category:
        clauses.append("question_category = ?")
        params.append(question_category)
    if difficulty:
        clauses.append("difficulty = ?")
        params.append(difficulty)
    if question_type:
        clauses.append("question_type = ?")
        params.append(question_type)
    if status and status != "all":
        clauses.append("status = ?")
        params.append(status)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM questions {where} ORDER BY RANDOM() LIMIT ? OFFSET ?",
        params + [limit, offset],
    ).fetchall()
    conn.close()

    return [_format_question(r) for r in rows]


@app.get("/questions/{question_id}")
def get_question(question_id: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM questions WHERE id=?", (question_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Question not found")
    return _format_question(row)


@app.post("/auth/request-otp")
def request_otp(body: OTPRequest):
    code = "".join(secrets.choice(string.digits) for _ in range(6))
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRY_MINUTES)).isoformat()
    conn = get_db()
    conn.execute("INSERT OR REPLACE INTO otps (email, code, expires_at) VALUES (?,?,?)",
                 (body.email.lower(), code, expires_at))
    conn.commit()
    conn.close()
    if RESEND_API_KEY or (GMAIL_USER and GMAIL_APP_PASSWORD):
        try:
            _send_otp_email(body.email.lower(), code)
        except Exception as e:
            print(f"[EMAIL ERROR] {e}")
            raise HTTPException(503, f"Could not send OTP email: {e}")
    else:
        print(f"[DEV] OTP for {body.email}: {code}")
    return {"message": "OTP sent"}


@app.post("/auth/verify-otp")
def verify_otp(body: OTPVerify):
    conn = get_db()
    email = body.email.lower()
    row = conn.execute("SELECT code, expires_at FROM otps WHERE email=?", (email,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(400, "No OTP found for this email")
    if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
        conn.close()
        raise HTTPException(400, "OTP expired")
    if row["code"] != body.code:
        conn.close()
        raise HTTPException(400, "Invalid OTP")

    # Get or create user
    user = conn.execute("SELECT id, signup_date FROM users WHERE email=?", (email,)).fetchone()
    if not user:
        user_id = str(uuid.uuid4())
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        conn.execute("INSERT INTO users (id, email, signup_date, created_at) VALUES (?,?,?,?)",
                     (user_id, email, today, datetime.now(timezone.utc).isoformat()))
        signup_date = today
    else:
        user_id = user["id"]
        signup_date = user["signup_date"]

    # Create session token
    token = secrets.token_urlsafe(32)
    conn.execute("INSERT INTO sessions (token, user_id, created_at) VALUES (?,?,?)",
                 (token, user_id, datetime.now(timezone.utc).isoformat()))
    conn.execute("DELETE FROM otps WHERE email=?", (email,))
    conn.commit()
    conn.close()

    return {"token": token, "user_id": user_id, "email": email, "signup_date": signup_date}


@app.post("/auth/logout")
def logout(x_session_token: Optional[str] = Header(default=None)):
    if x_session_token:
        conn = get_db()
        conn.execute("DELETE FROM sessions WHERE token=?", (x_session_token,))
        conn.commit()
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


@app.post("/attempts", response_model=AttemptOut)
def record_attempt(body: AttemptIn, x_session_token: Optional[str] = Header(default=None)):
    conn = get_db()
    user = _get_user_from_token(x_session_token, conn)
    q = conn.execute("SELECT * FROM questions WHERE id=?", (body.question_id,)).fetchone()
    if not q:
        conn.close()
        raise HTTPException(404, "Question not found")

    is_correct = body.chosen.upper() == q["correct"].upper()
    attempt_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Mark as daily if this question is in today's scheduled daily set
    daily_row = conn.execute("SELECT question_ids FROM daily_sets WHERE date=?", (today,)).fetchone()
    today_ids = set(json.loads(daily_row["question_ids"])) if daily_row else set()
    is_daily = 1 if body.question_id in today_ids else 0

    conn.execute(
        "INSERT INTO attempts (id, question_id, chosen, is_correct, time_taken, attempted_at, user_id, is_daily) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (attempt_id, body.question_id, body.chosen, int(is_correct), body.time_taken, now,
         user["id"] if user else None, is_daily),
    )
    conn.commit()
    conn.close()

    return AttemptOut(
        id=attempt_id,
        question_id=body.question_id,
        chosen=body.chosen,
        is_correct=is_correct,
        correct_answer=q["correct"],
        explanation=q["explanation"] or "",
    )


@app.get("/report")
def get_report(x_session_token: Optional[str] = Header(default=None)):
    conn = get_db()
    user = _get_user_from_token(x_session_token, conn)

    if not user:
        conn.close()
        return {
            "total_attempts": 0,
            "overall_accuracy_pct": 0,
            "streak_days": 0,
            "subject_breakdown": [
                {"subject": s, "total": 0, "correct": 0, "accuracy_pct": None, "status": "not_attempted"}
                for s in STATIC_SUBJECTS
            ],
            "weak_areas": [],
        }

    uid = user["id"]

    total_attempts = conn.execute(
        "SELECT COUNT(*) FROM attempts a WHERE a.user_id = ?", (uid,)
    ).fetchone()[0]
    correct_attempts = conn.execute(
        "SELECT COUNT(*) FROM attempts a WHERE a.user_id = ? AND is_correct=1", (uid,)
    ).fetchone()[0]

    attempted_rows = conn.execute("""
        SELECT COALESCE(q.upsc_subject, q.subject, 'Unknown') AS subject,
               COUNT(a.id)                     AS total,
               SUM(a.is_correct)               AS correct,
               ROUND(AVG(a.is_correct)*100, 1) AS accuracy_pct
        FROM attempts a
        JOIN questions q ON a.question_id = q.id
        WHERE a.user_id = ?
        GROUP BY subject
    """, (uid,)).fetchall()

    attempted_map = {r["subject"]: r for r in attempted_rows}

    subject_breakdown = []
    for subj in STATIC_SUBJECTS:
        if subj in attempted_map:
            r = attempted_map[subj]
            subject_breakdown.append({
                "subject": subj,
                "total": r["total"],
                "correct": r["correct"],
                "accuracy_pct": r["accuracy_pct"],
                "status": "attempted",
            })
        else:
            subject_breakdown.append({
                "subject": subj,
                "total": 0,
                "correct": 0,
                "accuracy_pct": None,
                "status": "not_attempted",
            })

    weak_areas = sorted(
        [s for s in subject_breakdown if s["status"] == "attempted" and s["total"] >= 2],
        key=lambda s: s["accuracy_pct"]
    )[:3]

    completed_days = conn.execute("""
        SELECT ds.date
        FROM daily_sets ds
        WHERE (
            SELECT COUNT(DISTINCT a.question_id)
            FROM attempts a
            WHERE a.question_id IN (SELECT value FROM json_each(ds.question_ids))
            AND DATE(a.attempted_at) = ds.date
            AND a.user_id = ?
        ) >= json_array_length(ds.question_ids)
        ORDER BY ds.date DESC
    """, (uid,)).fetchall()

    streak = 0
    prev = None
    for row in completed_days:
        d = row["date"]
        if prev is None or (datetime.fromisoformat(prev) - datetime.fromisoformat(d)).days == 1:
            streak += 1
            prev = d
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


@app.get("/daily")
def get_daily(
    date: Optional[str] = None,
    x_session_token: Optional[str] = Header(default=None),
):
    """
    Return today's 10-question set.
    The daily pipeline owns set creation — this endpoint reads it.
    Pass ?date=YYYY-MM-DD to fetch a past set (archive).
    Returns 503 if today's set hasn't been generated yet.
    """
    target_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    is_today = (target_date == datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    conn = get_db()

    row = conn.execute(
        "SELECT question_ids FROM daily_sets WHERE date=?", (target_date,)
    ).fetchone()

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

    # Fetch full question objects in order
    questions = []
    for q_id in q_ids:
        qrow = conn.execute("SELECT * FROM questions WHERE id=?", (q_id,)).fetchone()
        if qrow:
            questions.append(_format_question(qrow))

    # Fetch user's attempts for these questions (daily attempts only)
    attempts_map = {}
    user_filter = "AND user_id = ?" if user else ""
    user_param = [user["id"]] if user else []
    for q_id in q_ids:
        a = conn.execute(
            f"SELECT chosen, is_correct, attempted_at FROM attempts "
            f"WHERE question_id=? AND is_daily=1 {user_filter} "
            f"ORDER BY attempted_at DESC LIMIT 1",
            [q_id] + user_param
        ).fetchone()
        if a:
            attempts_map[q_id] = {"chosen": a["chosen"], "is_correct": bool(a["is_correct"]),
                                   "attempted_at": a["attempted_at"]}

    conn.close()

    attempted = len(attempts_map)
    correct   = sum(1 for a in attempts_map.values() if a["is_correct"])

    return {
        "date": target_date,
        "questions": questions,
        "attempts": attempts_map,
        "summary": {
            "total": len(questions),
            "attempted": attempted,
            "correct": correct,
            "completed": attempted == len(questions),
        }
    }


@app.get("/daily/archive")
def get_archive(
    detail: Optional[str] = None,   # ?detail=YYYY-MM-DD to get full question+attempt data for one set
    x_session_token: Optional[str] = Header(default=None),
):
    """
    List daily sets available to this user (from signup date onward).
    Each set includes state: 'attempted' | 'unattempted'.
    Pass ?detail=YYYY-MM-DD to get full question+attempt data for a specific set.
    """
    conn = get_db()
    user = _get_user_from_token(x_session_token, conn)
    signup_date = user["signup_date"] if user else None
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # If detail requested, return full question+attempt data for that date
    if detail:
        row = conn.execute("SELECT question_ids FROM daily_sets WHERE date=?", (detail,)).fetchone()
        if not row:
            conn.close()
            raise HTTPException(404, f"No set found for {detail}")
        q_ids = json.loads(row["question_ids"])
        user_filter = "AND user_id = ?" if user else ""
        user_param = [user["id"]] if user else []

        questions = []
        attempts_map = {}
        for q_id in q_ids:
            qrow = conn.execute("SELECT * FROM questions WHERE id=?", (q_id,)).fetchone()
            if qrow:
                questions.append(_format_question(qrow))
            a = conn.execute(
                f"SELECT chosen, is_correct, attempted_at FROM attempts "
                f"WHERE question_id=? {user_filter} ORDER BY attempted_at DESC LIMIT 1",
                [q_id] + user_param
            ).fetchone()
            if a:
                attempts_map[q_id] = {"chosen": a["chosen"], "is_correct": bool(a["is_correct"]),
                                       "attempted_at": a["attempted_at"]}
        conn.close()
        attempted_count = len(attempts_map)
        state = "attempted" if attempted_count > 0 else "unattempted"
        return {
            "date": detail,
            "state": state,
            "questions": questions,
            "attempts": attempts_map,
            "summary": {
                "total": len(questions),
                "attempted": attempted_count,
                "correct": sum(1 for a in attempts_map.values() if a["is_correct"]),
                "completed": attempted_count == len(questions),
            }
        }

    query = "SELECT date, question_ids, created_at FROM daily_sets"
    params = []
    if signup_date:
        query += " WHERE date >= ?"
        params.append(signup_date)
    query += " ORDER BY date DESC"

    sets = conn.execute(query, params).fetchall()

    result = []
    for s in sets:
        if s["date"] == today:
            continue  # today is shown via /daily, not archive
        q_ids = json.loads(s["question_ids"])
        user_filter = "AND user_id = ?" if user else ""
        user_param = [user["id"]] if user else []

        attempted = conn.execute(
            f"SELECT COUNT(DISTINCT question_id) FROM attempts "
            f"WHERE question_id IN ({','.join('?' * len(q_ids))}) {user_filter}",
            q_ids + user_param
        ).fetchone()[0]
        correct = conn.execute(
            f"SELECT COUNT(*) FROM attempts "
            f"WHERE question_id IN ({','.join('?' * len(q_ids))}) AND is_correct=1 {user_filter}",
            q_ids + user_param
        ).fetchone()[0]
        state = "attempted" if attempted > 0 else "unattempted"
        result.append({
            "date": s["date"],
            "total": len(q_ids),
            "attempted": attempted,
            "correct": correct,
            "completed": attempted == len(q_ids),
            "state": state,
        })

    conn.close()
    return result


@app.get("/streak/calendar")
def get_streak_calendar(x_session_token: Optional[str] = Header(default=None)):
    """
    Return 7-day window centred on today (today = position 4).
    Each day: date, attempted (bool), is_today (bool).
    """
    conn = get_db()
    user = _get_user_from_token(x_session_token, conn)
    today = datetime.now(timezone.utc).date()

    # Build 7-day window: today is index 3 (0-based), i.e. positions -3 to +3
    dates = [today + timedelta(days=i) for i in range(-3, 4)]

    user_filter = "AND user_id = ?" if user else ""
    user_param = [user["id"]] if user else []

    # A day is "completed" (streak-worthy) if user answered ALL questions in that day's set on that day
    completed_dates = set()
    streak_rows = conn.execute(f"""
        SELECT ds.date
        FROM daily_sets ds
        WHERE (
            SELECT COUNT(DISTINCT a.question_id)
            FROM attempts a
            WHERE a.question_id IN (SELECT value FROM json_each(ds.question_ids))
            AND DATE(a.attempted_at) = ds.date
            {"AND a.user_id = ?" if user else ""}
        ) >= json_array_length(ds.question_ids)
    """, ([user["id"]] if user else [])).fetchall()
    completed_dates = {r["date"] for r in streak_rows}

    # A day is "partially attempted" if user answered some (but not all) questions
    partial_rows = conn.execute(f"""
        SELECT ds.date
        FROM daily_sets ds
        WHERE (
            SELECT COUNT(DISTINCT a.question_id)
            FROM attempts a
            WHERE a.question_id IN (SELECT value FROM json_each(ds.question_ids))
            AND DATE(a.attempted_at) = ds.date
            {"AND a.user_id = ?" if user else ""}
        ) > 0
        AND ds.date NOT IN ({",".join("?" * len(completed_dates)) if completed_dates else "NULL"})
    """, ([user["id"]] if user else []) + list(completed_dates)).fetchall()
    partial_dates = {r["date"] for r in partial_rows}
    conn.close()

    return {
        "today": today.isoformat(),
        "days": [
            {
                "date": d.isoformat(),
                "completed": d.isoformat() in completed_dates,      # full set done → streak
                "partial": d.isoformat() in partial_dates,           # some questions done
                "is_today": d == today,
                "is_future": d > today,
            }
            for d in dates
        ]
    }


@app.get("/subjects")
def list_subjects():
    conn = get_db()
    rows = conn.execute(
        "SELECT subject, COUNT(*) as count FROM questions GROUP BY subject ORDER BY count DESC"
    ).fetchall()
    conn.close()
    return [{"subject": r["subject"], "count": r["count"]} for r in rows]


@app.post("/daily/restart")
def restart_daily(date: Optional[str] = None):
    """Clear all attempts for today's daily set so user can retake."""
    target_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = get_db()

    row = conn.execute(
        "SELECT question_ids FROM daily_sets WHERE date=?", (target_date,)
    ).fetchone()

    if not row:
        conn.close()
        raise HTTPException(404, "No daily set found for this date")

    q_ids = json.loads(row["question_ids"])
    placeholders = ",".join("?" * len(q_ids))
    deleted = conn.execute(
        f"DELETE FROM attempts WHERE question_id IN ({placeholders})",
        q_ids
    ).rowcount
    conn.commit()
    conn.close()
    return {"date": target_date, "attempts_cleared": deleted}


@app.post("/internal/sync")
def internal_sync():
    """Trigger a DB sync + re-import from JSON files without restarting."""
    inserted, skipped = import_questions()
    updated = sync_statuses()
    return {"inserted": inserted, "skipped": skipped, "statuses_synced": updated}


class PushDailySetIn(BaseModel):
    date: str               # YYYY-MM-DD
    question_ids: List[str]
    secret: str


@app.post("/internal/push-daily-set")
def push_daily_set(body: PushDailySetIn):
    """
    Called by the Mac pipeline after generating today's set.
    Writes the daily_set record so Railway API serves it immediately.
    Protected by PIPELINE_SECRET env var.
    """
    if PIPELINE_SECRET and body.secret != PIPELINE_SECRET:
        raise HTTPException(403, "Forbidden")
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO daily_sets (date, question_ids, created_at) VALUES (?,?,?)",
        (body.date, json.dumps(body.question_ids), datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()
    return {"date": body.date, "questions": len(body.question_ids)}


@app.get("/stats")
def stats():
    conn = get_db()
    total_q = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    passed_q = conn.execute("SELECT COUNT(*) FROM questions WHERE status='pass'").fetchone()[0]
    total_a = conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    conn.close()
    return {"total_questions": total_q, "passed_questions": passed_q, "total_attempts": total_a}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_question(row) -> dict:
    sr_raw = row["suggested_reading"] if "suggested_reading" in row.keys() else None
    suggested_reading = json.loads(sr_raw) if sr_raw else None
    return {
        "id": row["id"],
        "question": row["question"],
        "options": json.loads(row["options"]),
        "correct_answer": row["correct"],
        "explanation": row["explanation"],
        "difficulty": row["difficulty"],
        "question_type": row["question_type"],
        "source_type": row["source_type"],
        "upsc_subject": row["upsc_subject"],
        "upsc_topic": row["upsc_topic"],
        "upsc_category": row["broad_category"],
        "question_category": row["question_category"],
        "suggested_reading": suggested_reading,
        "is_pyq": "pyq" in (row["source_file"] or "").lower(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=PORT, reload=True)
