from __future__ import annotations

from typing import List

from .memory_store import MemoryStore
from .compression import MemoryCompressor


class MemoryUpdater:
    """Maintain size of the memory table with lightweight summarisation."""

    def __init__(self, store: MemoryStore, max_size: int = 1000) -> None:
        self.memory_store = store
        self.max_size = max_size
        self.compressor = MemoryCompressor()

    def post_conversation_update(self, conversation_log: str) -> None:
        """Entry point after a conversation exchange."""
        self.update_access_counts(conversation_log)
        if len(self.memory_store.get_all()) > self.max_size:
            self._compress_old_memories()

    # ------------------------------------------------------------------
    # stubs for future entity/solution extraction
    # ------------------------------------------------------------------
    def extract_entities(self, log: str) -> List[str]:
        return []

    def extract_solutions(self, log: str) -> List[str]:
        return []

    def update_access_counts(self, log: str) -> None:  # pragma: no cover - stub
        pass

    # ------------------------------------------------------------------
    # compaction logic
    # ------------------------------------------------------------------
    def _compress_old_memories(self) -> None:
        """Summarise and delete the least important memories."""
        memories = self.memory_store.get_all()
        if len(memories) <= self.max_size:
            return

        sorted_mems = sorted(
            memories.values(), key=lambda m: (m.importance_weight, m.timestamp)
        )
        k = len(memories) - self.max_size + 1
        to_summarise = sorted_mems[:k]

        combined = "\n".join(m.content for m in to_summarise)
        summary = self.compressor.compress_text(combined)

        cur = self.memory_store.conn.cursor()
        for mem in to_summarise:
            cur.execute("DELETE FROM memory_fragments WHERE mem_id=?", (mem.memory_id,))
        self.memory_store.conn.commit()

        self.memory_store.add(summary, importance=0.8, source_type="summary")

