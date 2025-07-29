import math
import os
from datetime import datetime, timezone
from typing import Dict, Any

import faiss

from .memory import Memory
from .token_counter import TokenCounter
from .vector_memory import VectorMemory


class RelevanceEngine:
    def __init__(self):
        self.token_counter = TokenCounter()
        self.vector_memory = VectorMemory()
        try:
            self.vector_memory.load()
        except Exception:
            self.vector_memory = None

    def _semantic_similarity(self, a: str, b: str) -> float:
        """Placeholder for semantic similarity."""
        return 0.5 if a and b else 0.0

    def score_all(
        self, memories: Dict[str, Memory], task: str, conversation_id: str
    ) -> Dict[str, Dict[str, Any]]:
        scores: Dict[str, Dict[str, Any]] = {}
        current_project_id = None
        current_entities = set()

        seen_texts = set()

        for memory_id, memory in memories.items():
            score = 0.0

            age_hours = (
                datetime.now(tz=timezone.utc) - memory.timestamp
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
            seen_texts.add(memory.content)

        # incorporate vector memory hits
        if self.vector_memory and task:
            hits = self.vector_memory.search(task, top_k=8)
            if hits:
                metric = getattr(self.vector_memory.index, "metric_type", None)
                for entry, dist in hits:
                    if entry.text in seen_texts:
                        continue
                    mem = Memory(
                        memory_id=entry.id,
                        content=entry.text,
                        timestamp=datetime.fromtimestamp(entry.timestamp, tz=timezone.utc),
                        type="vector",  # type: ignore[str]
                        project_id=None,
                        entities=set(),
                        importance_weight=1.0,
                        access_count=1,
                    )
                    similarity = dist
                    if metric == faiss.METRIC_L2:
                        similarity = 1.0 - dist
                    similarity = max(0.0, min(float(similarity), 1.0))
                    vec_score = 20 * similarity
                    scores[mem.memory_id] = {
                        "memory": mem,
                        "score": vec_score,
                        "token_cost": self.token_counter.count(mem.content),
                    }
                    seen_texts.add(mem.content)

        return scores
