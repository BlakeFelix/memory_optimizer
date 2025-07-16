import re
from typing import List

from .memory_store import MemoryStore


class MemoryUpdater:
    def __init__(self, store: MemoryStore, max_size: int = 1000):
        self.memory_store = store
        self.max_size = max_size

    def post_conversation_update(self, conversation_log: str) -> None:
        entities = self.extract_entities(conversation_log)
        solutions = self.extract_solutions(conversation_log)
        self.update_access_counts(conversation_log)
        for sol in solutions:
            self.memory_store.add(sol)
        if len(self.memory_store.get_all()) > self.max_size:
            self.compress_old_memories()

    def extract_entities(self, log: str) -> List[str]:
        """Extract a variety of common entities using regex only."""
        patterns = {
            "email": r"[\w\.-]+@[\w\.-]+",
            "phone": r"\b\+?\d[\d\s()-]{7,}\b",
            "url": r"https?://\S+",
            "money": r"\$\d+(?:\.\d+)?",
            "date": r"\b\d{4}-\d{2}-\d{2}\b",
            "time": r"\b\d{1,2}:\d{2}(?:am|pm)?\b",
            "person": r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b",
            "single_word": r"\b[A-Z][a-zA-Z]+\b",
            "ip": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "hashtag": r"#\w+",
            "mention": r"@\w+",
        }

        entities = []
        for patt in patterns.values():
            entities.extend(re.findall(patt, log))

        flow = re.findall(r"(user|assistant):", log, re.I)
        if len(set(flow)) > 1:
            entities.append("conversation_flow")

        cleaned = [e.strip(".,") for e in entities]
        return list(set(cleaned))

    def extract_solutions(self, log: str) -> List[str]:
        pairs = []
        for match in re.finditer(r"error:(.*?)solution:(.*?)(?:\n|$)", log, re.I | re.S):
            content = f"Error: {match.group(1).strip()} Solution: {match.group(2).strip()}"
            pairs.append(content)
        memories = []
        from .memory import Memory
        from datetime import datetime
        for idx, txt in enumerate(pairs):
            memories.append(
                Memory(
                    memory_id=f"errsol-{int(datetime.now().timestamp())}-{idx}",
                    content=txt,
                    timestamp=datetime.now(),
                    type="error_solution",
                )
            )
        return memories

    def update_access_counts(self, log: str) -> None:
        for mem in self.memory_store.get_all().values():
            if mem.content and mem.content[:50] in log:
                mem.access_count += 1
                self.memory_store.add(mem)

    def compress_old_memories(self) -> None:
        memories = list(self.memory_store.get_all().values())
        memories.sort(key=lambda m: m.timestamp)
        while len(memories) > self.max_size:
            old = memories.pop(0)
            del self.memory_store.memories[old.memory_id]
