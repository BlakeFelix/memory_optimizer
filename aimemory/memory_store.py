import os
from typing import Dict, List
from .memory import Memory


class MemoryStore:
    """Hierarchical memory storage with simple in-memory index."""

    def __init__(self, base_path: str):
        self.base_path = os.path.expanduser(base_path)
        os.makedirs(self.base_path, exist_ok=True)
        self.memories: Dict[str, Memory] = {}

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

    def get_all(self) -> Dict[str, Memory]:
        return self.memories
