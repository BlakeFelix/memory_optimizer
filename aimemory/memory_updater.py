from typing import List

from .memory_store import MemoryStore


class MemoryUpdater:
    def __init__(self, store: MemoryStore, max_size: int = 1000):
        self.memory_store = store
        self.max_size = max_size

    def post_conversation_update(self, conversation_log: str) -> None:
        # placeholder implementation
        self.update_access_counts(conversation_log)
        if len(self.memory_store.get_all()) > self.max_size:
            self.compress_old_memories()

    def extract_entities(self, log: str) -> List[str]:
        return []

    def extract_solutions(self, log: str) -> List[str]:
        return []

    def update_access_counts(self, log: str) -> None:
        pass

    def compress_old_memories(self) -> None:
        pass
