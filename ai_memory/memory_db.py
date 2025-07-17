"""
AI-Memory SQLite backend
------------------------
* Normalised 5-table schema
* WAL + pragmatic PRAGMAs tuned for local NVMe
* JSON-importer with entity extraction
* No third-party dependencies

Schema overview
---------------
  users                <- single-row table for now (future multi-user)
  conversations        <- high-level chat sessions
  messages             <- every assistant / user line
  entities             <- canonicalised entities (person, date, url...)
  memory_fragments     <- compressed chunks used by the optimiser
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

# ---------------------------------------------------------------------
#  CONFIG
# ---------------------------------------------------------------------

ROOT = Path(os.getenv("AI_MEMORY_ROOT", "~/ai_memory")).expanduser()
DB_PATH = ROOT / "ai_memory.db"
LEGACY_JSON_ROOTS: List[Path] = [
    ROOT,
    Path("~/aimemory").expanduser(),
]

# ---------------------------------------------------------------------
#  CONNECTION / SCHEMA
# ---------------------------------------------------------------------


def _connect() -> sqlite3.Connection:
    ROOT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist; idempotent."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id     TEXT PRIMARY KEY,
            name        TEXT,
            meta        TEXT
        );

        CREATE TABLE IF NOT EXISTS conversations (
            conv_id     TEXT PRIMARY KEY,
            user_id     TEXT REFERENCES users(user_id),
            title       TEXT,
            started_at  TEXT,
            updated_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS messages (
            msg_id      TEXT PRIMARY KEY,
            conv_id     TEXT REFERENCES conversations(conv_id),
            role        TEXT,
            content     TEXT,
            timestamp   TEXT
        );

        CREATE TABLE IF NOT EXISTS entities (
            entity_id   TEXT PRIMARY KEY,
            type        TEXT,
            value       TEXT,
            canonical   TEXT
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_value
            ON entities(type, canonical);

        CREATE TABLE IF NOT EXISTS message_entities (
            msg_id      TEXT REFERENCES messages(msg_id),
            entity_id   TEXT REFERENCES entities(entity_id),
            PRIMARY KEY (msg_id, entity_id)
        );

        CREATE INDEX IF NOT EXISTS idx_message_entities_entity
            ON message_entities(entity_id);

        CREATE TABLE IF NOT EXISTS memory_fragments (
            mem_id          TEXT PRIMARY KEY,
            conv_id         TEXT REFERENCES conversations(conv_id),
            msg_id          TEXT REFERENCES messages(msg_id),
            content         TEXT,
            importance      REAL,
            source_type     TEXT,
            token_estimate  INTEGER,
            created_at      TEXT
        );
        """
    )

    # ------------------------------------------------------------------
    # ensure new columns exist when upgrading old DBs
    # ------------------------------------------------------------------
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(memory_fragments)")
    cols = [row[1] for row in cur.fetchall()]
    if "source_type" not in cols:
        conn.execute("ALTER TABLE memory_fragments ADD COLUMN source_type TEXT")


@contextmanager
def _db() -> Iterable[sqlite3.Connection]:
    conn = _connect()
    try:
        _ensure_schema(conn)
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------
#  SIMPLE TOKEN ESTIMATOR (~85% accurate, 0 deps)
# ---------------------------------------------------------------------


def rough_token_len(text: str) -> int:
    """Approximate tokens: 1 token ~= 4 chars; tweak for URLs / code / emojis."""
    base = len(text) // 4
    bonus = text.count("http") + text.count("https")
    bonus += text.count("\n")
    bonus += len(re.findall(r"[^\w\s]", text)) // 10
    return base + bonus


# ---------------------------------------------------------------------
#  REGEX ENTITY EXTRACTOR
# ---------------------------------------------------------------------

_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")),
    ("url", re.compile(r"https?://[^\s]+")),
    ("phone", re.compile(r"\b\+?\d[\d\-\s]{7,}\b")),
    ("money", re.compile(r"\$\s?\d+(?:,\d{3})*(?:\.\d{2})?")),
    ("date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("person", re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")),
]


def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Return list of (type, value). Simple regex - no NLP libs."""
    found: List[Tuple[str, str]] = []
    for etype, pat in _PATTERNS:
        for match in pat.findall(text):
            found.append((etype, match))
    return found


# ---------------------------------------------------------------------
#  IMPORTER FROM JSON MEMORY FILES
# ---------------------------------------------------------------------


UNWANTED_SEGMENTS = {"venv", "node_modules", "site-packages", "__pycache__"}


def _load_json_files() -> Iterable[Path]:
    for root in LEGACY_JSON_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if any(seg in UNWANTED_SEGMENTS for seg in path.parts):
                continue
            try:
                if path.stat().st_size > 1_000_000:
                    continue
            except OSError as exc:
                print(f"[IMPORT] skipping {path}: {exc}")
                continue
            yield path


def _canonical(term: str) -> str:
    return term.lower().strip()


def import_legacy_json() -> None:
    """Walk both the legacy and new JSON trees, load into SQLite."""
    with _db() as conn:
        cur = conn.cursor()
        user_id = "default"
        cur.execute(
            "INSERT OR IGNORE INTO users (user_id, name) VALUES (?,?)",
            (user_id, "default_user"),
        )

        for json_path in _load_json_files():
            try:
                raw = json.loads(json_path.read_text())
                content = raw.get("content") or raw.get("text") or raw.get("message") or ""
                if not content:
                    continue

                msg_id = raw.get("id") or str(uuid.uuid4())
                ts = raw.get("timestamp") or datetime.now(tz=timezone.utc).isoformat()
                conv_id = raw.get("conversation_id") or raw.get("project_id") or "import"

                cur.execute(
                    "INSERT OR IGNORE INTO conversations (conv_id, user_id, title, started_at, updated_at) VALUES (?,?,?,?,?)",
                    (conv_id, user_id, raw.get("type", "imported"), ts, ts),
                )

                cur.execute(
                    "INSERT OR IGNORE INTO messages (msg_id, conv_id, role, content, timestamp) VALUES (?,?,?,?,?)",
                    (msg_id, conv_id, "system", content, ts),
                )

                for etype, value in extract_entities(content):
                    canonical = _canonical(value)
                    entity_id = f"{etype}:{canonical}"
                    cur.execute(
                        "INSERT OR IGNORE INTO entities (entity_id, type, value, canonical) VALUES (?,?,?,?)",
                        (entity_id, etype, value, canonical),
                    )
                    cur.execute(
                        "INSERT OR IGNORE INTO message_entities (msg_id, entity_id) VALUES (?,?)",
                        (msg_id, entity_id),
                    )

                cur.execute(
                    "INSERT OR IGNORE INTO memory_fragments (mem_id, conv_id, msg_id, content, importance, token_estimate, created_at, source_type) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        raw.get("id", msg_id),
                        conv_id,
                        msg_id,
                        content,
                        float(raw.get("importance_weight", 1.0)),
                        rough_token_len(content),
                        ts,
                        raw.get("source_type") or raw.get("type") or "conversation",
                    ),
                )
            except Exception as exc:
                print(f"[IMPORT] skipping {json_path}: {exc}")
                continue
        conn.commit()
    print("[IMPORT] legacy JSON import completed.")


# ---------------------------------------------------------------------
#  PUBLIC CONSTRUCTOR (for Claude spec)
# ---------------------------------------------------------------------


def create_production_memory_system(config_file: str | None = None, skip_import: bool = False):
    """Return a connection ready for higher-level optimizers."""
    first_init = not DB_PATH.exists()
    conn = _connect()
    _ensure_schema(conn)

    if first_init and not skip_import:
        import_legacy_json()

    print(f"[AI-Memory] SQLite backend ready -> {DB_PATH}")
    return conn


# ---------------------------------------------------------------------
#  Smoke test when run directly
# ---------------------------------------------------------------------

if __name__ == "__main__":
    conn = create_production_memory_system()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM memory_fragments")
    print("Fragments in DB:", cur.fetchone()[0])
    conn.close()
