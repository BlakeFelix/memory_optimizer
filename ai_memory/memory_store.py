from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .memory import Memory

from .memory_db import (
    _ensure_schema,
    rough_token_len,
    extract_entities,
)


def _db_path() -> str:
    """Return current SQLite DB path, creating directories if needed."""
    root = Path(os.getenv("AI_MEMORY_ROOT", "~/ai_memory")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return str(root / "ai_memory.db")


class MemoryStore:
    """Thin wrapper around the SQLite backend."""

    def __init__(self, conn: Optional[sqlite3.Connection] = None) -> None:
        if conn is None:
            self.conn = sqlite3.connect(_db_path(), check_same_thread=False)
            _ensure_schema(self.conn)
        else:
            self.conn = conn

    def add(
        self,
        content: str,
        conv_id: Optional[str] = None,
        msg_id: Optional[str] = None,
        importance: float = 1.0,
        source_type: str = "conversation",
    ) -> str:
        mem_id = str(uuid.uuid4())
        ts = datetime.now(tz=timezone.utc).isoformat()
        cur = self.conn.cursor()

        if msg_id is None:
            msg_id = mem_id

        cur.execute(
            "INSERT OR IGNORE INTO messages (msg_id, conv_id, role, content, timestamp) VALUES (?,?,?,?,?)",
            (msg_id, conv_id, "system", content, ts),
        )

        for etype, value in extract_entities(content):
            canonical = value.lower().strip()
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
            """
            INSERT INTO memory_fragments
                (mem_id, conv_id, msg_id, content, importance, token_estimate, created_at, source_type, access_count)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                mem_id,
                conv_id,
                msg_id,
                content,
                importance,
                rough_token_len(content),
                ts,
                source_type,
                0,
            ),
        )
        self.conn.commit()
        return mem_id

    def update_access(self, mem_id: str) -> None:
        ts = datetime.now(tz=timezone.utc).isoformat()
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE memory_fragments SET created_at=?, access_count=access_count+1 WHERE mem_id=?",
            (ts, mem_id),
        )
        self.conn.commit()

    def get_all(self) -> Dict[str, Memory]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT mem_id, conv_id, content, importance, created_at, source_type, access_count FROM memory_fragments"
        )
        memories: Dict[str, Memory] = {}
        for mem_id, conv_id, content, importance, created_at, source_type, access_count in cur.fetchall():
            try:
                ts = datetime.fromisoformat(created_at)
            except Exception:
                ts = datetime.now(tz=timezone.utc)
            memories[mem_id] = Memory(
                memory_id=mem_id,
                content=content,
                timestamp=ts,
                type=source_type,
                project_id=conv_id,
                importance_weight=float(importance),
                entities=set(),
                access_count=int(access_count or 0),
            )
        return memories
