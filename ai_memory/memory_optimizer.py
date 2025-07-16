from typing import Dict, Any

from .memory_store import MemoryStore
from .relevance_engine import RelevanceEngine
from .context_builder import ContextBuilder


class MemoryOptimizer:
    def __init__(self, memory_path: str = "~/ai_memory/"):
        self.store = MemoryStore()
        self.engine = RelevanceEngine()
        self.builder = ContextBuilder()

    def _budget(self, spec: Dict[str, Any]) -> int:
        return int(spec.get("max_tokens", 0))

    def build_optimal_context(
        self,
        model_spec: Dict[str, Any],
        current_task: str | None = None,
        conversation_id: str | None = None,
    ) -> str:
        scored = self.engine.score_all(
            current_task=current_task,
            conv_id=conversation_id,
        )
        return self.builder.build_layers(
            scored_memories=scored,
            token_budget=self._budget(model_spec),
        )
