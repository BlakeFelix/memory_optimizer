import time
from .memory_optimizer import MemoryOptimizer
from .memory_updater import MemoryUpdater
from datetime import datetime, timezone


class ConversationSession:
    def __init__(self):
        self.session_id = None
        self.start_ts = time.time()
        self.title = "session"
        self.optimizer = MemoryOptimizer()
        self.updater = MemoryUpdater(self.optimizer.memory_store)

    def add_exchange(self, user_msg: str, assistant_msg: str) -> None:
        """Record a chat exchange and trigger maintenance."""
        cur = self.optimizer.memory_store.conn.cursor()
        ts = datetime.now(tz=timezone.utc).isoformat()
        if self.session_id:
            cur.execute(
                "INSERT OR IGNORE INTO conversations (conv_id, user_id, title, started_at, updated_at) VALUES (?,?,?,?,?)",
                (self.session_id, "default", self.title, ts, ts),
            )
            cur.execute(
                "UPDATE conversations SET updated_at=? WHERE conv_id=?",
                (ts, self.session_id),
            )
        self.optimizer.memory_store.add(user_msg, conv_id=self.session_id)
        self.optimizer.memory_store.add(assistant_msg, conv_id=self.session_id)
        log = f"{user_msg}\n{assistant_msg}"
        self.updater.post_conversation_update(log)

    def build_context(self, query: str, model: str = "gpt-4", limit: int | None = None) -> str:
        max_tokens = limit if limit is not None else 4096
        return self.optimizer.build_optimal_context(
            {"name": model, "max_tokens": max_tokens},
            current_task=query,
            conversation_id=self.session_id,
        )

