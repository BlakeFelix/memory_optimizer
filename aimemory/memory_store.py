import os
import sqlite3
import shutil
from threading import Lock
from typing import Dict, List
from datetime import datetime

from .memory import Memory


class MemoryStore:
    """Hierarchical memory storage with simple in-memory index."""

    def __init__(self, base_path: str):
        self.base_path = os.path.expanduser(base_path)
        os.makedirs(self.base_path, exist_ok=True)
        self.db_path = os.path.join(self.base_path, "memories.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.lock = Lock()
        self._create_table()
        self.memories: Dict[str, Memory] = {}
        self._load()

    def _create_table(self) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT,
                    timestamp TEXT,
                    type TEXT,
                    project_id TEXT,
                    entities TEXT,
                    importance REAL,
                    access_count INTEGER
                )
                """
            )
            self.conn.commit()

    def _load(self) -> None:
        with self.lock:
            cur = self.conn.cursor()
            for row in cur.execute("SELECT * FROM memories"):
                mem = Memory(
                    memory_id=row[0],
                    content=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    type=row[3],
                    project_id=row[4],
                    entities=set(row[5].split("|")) if row[5] else set(),
                    importance_weight=row[6],
                    access_count=row[7],
                )
                self.memories[mem.memory_id] = mem

        # Predefined structure placeholder
        self.structure = {
            "identity": {"core_traits": [], "entities": [], "preferences": []},
            "projects": {"active": {}, "archived": {}, "code_patterns": {}},
            "conversations": {
                "summaries": [],
                "key_exchanges": [],
                "error_solutions": {},
            },
            "knowledge": {"technical": {}, "domain": {}, "meta": {}},
        }

    def add(self, memory: Memory) -> None:
        self.memories[memory.memory_id] = memory
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO memories
                (memory_id, content, timestamp, type, project_id, entities, importance, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.memory_id,
                    memory.content,
                    memory.timestamp.isoformat(),
                    memory.type,
                    memory.project_id,
                    "|".join(memory.entities),
                    memory.importance_weight,
                    memory.access_count,
                ),
            )
            self.conn.commit()
        self._backup()

    def close(self) -> None:
        with self.lock:
            self.conn.commit()
            self.conn.close()

    def get_all(self) -> Dict[str, Memory]:
        return self.memories

    def _backup(self) -> None:
        """Create a simple backup of the database."""
        backup_path = self.db_path + ".bak"
        try:
            shutil.copy2(self.db_path, backup_path)
        except Exception:
            pass
