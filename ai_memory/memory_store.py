from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .memory_db import create_production_memory_system, rough_token_len


class MemoryStore:
    """Thin wrapper around the SQLite backend."""

    def __init__(self, conn: Optional[sqlite3.Connection] = None) -> None:
        self.conn = conn or create_production_memory_system()

    def add(
        self,
        content: str,
        conv_id: Optional[str] = None,
        msg_id: Optional[str] = None,
        importance: float = 1.0,
    ) -> str:
        mem_id = str(uuid.uuid4())
        ts = datetime.now(tz=timezone.utc).isoformat()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO memory_fragments
                (mem_id, conv_id, msg_id, content, importance, token_estimate, created_at)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                mem_id,
                conv_id,
                msg_id,
                content,
                importance,
                rough_token_len(content),
                ts,
            ),
        )
        self.conn.commit()
        return mem_id

    def update_access(self, mem_id: str) -> None:
        ts = datetime.now(tz=timezone.utc).isoformat()
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE memory_fragments SET created_at=? WHERE mem_id=?",
            (ts, mem_id),
        )
        self.conn.commit()

    def get_all(self) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT mem_id, conv_id, msg_id, content, importance, token_estimate, created_at FROM memory_fragments"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
