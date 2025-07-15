import math
from datetime import datetime
from functools import lru_cache
from typing import Dict, Any

from .memory import Memory
from .token_counter import TokenCounter

try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore
    _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:  # pragma: no cover
    _EMBED_MODEL = None
    from difflib import SequenceMatcher


class RelevanceEngine:
    def __init__(self):
        self.token_counter = TokenCounter()

    @lru_cache(maxsize=1024)
    def _embed(self, text: str):
        if _EMBED_MODEL:
            return _EMBED_MODEL.encode(text)
        return text

    def _semantic_similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        if _EMBED_MODEL:
            va = self._embed(a)
            vb = self._embed(b)
            return float(util.cos_sim(va, vb))
        return SequenceMatcher(None, a, b).ratio()

    def score_all(
        self, memories: Dict[str, Memory], task: str, conversation_id: str
    ) -> Dict[str, Dict[str, Any]]:
        scores: Dict[str, Dict[str, Any]] = {}
        current_project_id = None
        current_entities = set()

        for memory_id, memory in memories.items():
            score = 0.0

            age_hours = (
                datetime.now() - memory.timestamp
            ).total_seconds() / 3600.0
            score += 10 * math.exp(-age_hours / 168)

            if task:
                score += 20 * self._semantic_similarity(memory.content, task)

            if current_project_id and memory.project_id == current_project_id:
                score += 15

            score += 5 * math.log1p(memory.access_count)

            score += 8 * len(memory.entities & current_entities)

            if memory.type == "error_solution":
                score += 12

            score += memory.importance_weight * 10

            scores[memory_id] = {
                "memory": memory,
                "score": score,
                "token_cost": self.token_counter.count(memory.content),
            }

        return scores
