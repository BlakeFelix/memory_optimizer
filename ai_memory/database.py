import os
import sqlite3
import hashlib
import json
from typing import Any, Dict

class MemoryDatabase:
    """SQLite backed database with normalized tables."""

    def __init__(self, db_path: str):
        self.db_path = os.path.expanduser(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT
            );

            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                user_id TEXT,
                started_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT,
                sender TEXT,
                content TEXT,
                timestamp TEXT,
                FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
            );

            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                type TEXT,
                value TEXT UNIQUE
            );

            CREATE TABLE IF NOT EXISTS memory_fragments (
                fragment_id TEXT PRIMARY KEY,
                message_id TEXT,
                entity_id TEXT,
                importance REAL,
                FOREIGN KEY(message_id) REFERENCES messages(message_id),
                FOREIGN KEY(entity_id) REFERENCES entities(entity_id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_messages_time ON messages(timestamp);
            CREATE INDEX IF NOT EXISTS idx_entities_value ON entities(value);
            CREATE INDEX IF NOT EXISTS idx_fragments_entity ON memory_fragments(entity_id);

            CREATE TRIGGER IF NOT EXISTS trg_delete_conversation
            AFTER DELETE ON conversations
            BEGIN
                DELETE FROM messages WHERE conversation_id = OLD.conversation_id;
            END;

            CREATE TRIGGER IF NOT EXISTS trg_delete_message
            AFTER DELETE ON messages
            BEGIN
                DELETE FROM memory_fragments WHERE message_id = OLD.message_id;
            END;
            """
        )
        self.conn.commit()

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def import_json(self, json_path: str) -> None:
        """Import conversation data from an aimemory JSON file."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - invalid input
            raise ValueError(f"Failed to load {json_path}: {exc}") from exc

        with self.conn:
            cur = self.conn.cursor()
            user_name = data.get("user", "user")
            user_id = self._hash(user_name)
            cur.execute(
                "INSERT OR IGNORE INTO users (user_id, name) VALUES (?, ?)",
                (user_id, user_name),
            )

            for conv in data.get("conversations", []):
                conv_id = self._hash(str(conv.get("id", json.dumps(conv))))
                cur.execute(
                    "INSERT OR IGNORE INTO conversations (conversation_id, user_id, started_at) VALUES (?, ?, ?)",
                    (conv_id, user_id, conv.get("started_at")),
                )

                for msg in conv.get("messages", []):
                    message_content = msg.get("content", "")
                    msg_id = self._hash(msg.get("id", message_content))
                    cur.execute(
                        "INSERT OR IGNORE INTO messages (message_id, conversation_id, sender, content, timestamp) VALUES (?, ?, ?, ?, ?)",
                        (
                            msg_id,
                            conv_id,
                            msg.get("sender"),
                            message_content,
                            msg.get("timestamp"),
                        ),
                    )

                    for ent in msg.get("entities", []):
                        if isinstance(ent, dict):
                            val = ent.get("value")
                            etype = ent.get("type")
                            importance = float(ent.get("importance", 0))
                        else:
                            val = str(ent)
                            etype = None
                            importance = 0.0

                        if not val:
                            continue

                        entity_id = self._hash(val)
                        cur.execute(
                            "INSERT OR IGNORE INTO entities (entity_id, type, value) VALUES (?, ?, ?)",
                            (entity_id, etype, val),
                        )
                        fragment_id = self._hash(msg_id + entity_id)
                        cur.execute(
                            "INSERT OR IGNORE INTO memory_fragments (fragment_id, message_id, entity_id, importance) VALUES (?, ?, ?, ?)",
                            (fragment_id, msg_id, entity_id, importance),
                        )

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

