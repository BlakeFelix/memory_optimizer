from typing import Dict, Any

from .memory_store import MemoryStore
from .relevance_engine import RelevanceEngine
from .token_counter import TokenCounter
from .context_builder import ContextBuilder


class MemoryOptimizer:
    def __init__(self) -> None:
        self.memory_store = MemoryStore()
        self.relevance_engine = RelevanceEngine()
        self.token_counter = TokenCounter()
        self.context_builder = ContextBuilder()

    def _calculate_token_budget(self, model_spec: Dict[str, Any]) -> int:
        return int(model_spec.get("max_tokens", 0))

    def build_optimal_context(
        self,
        model_spec: Dict[str, Any],
        current_task: str = None,
        conversation_id: str = None,
    ) -> str:
        budget = self._calculate_token_budget(model_spec)

        scored = self.relevance_engine.score_all(
            memories=self.memory_store.get_all(),
            task=current_task,
            conversation_id=conversation_id,
        )

        return self.context_builder.build_layers(
            scored_memories=scored, token_budget=budget
        )
