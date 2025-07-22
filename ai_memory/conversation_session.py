import time
from .memory_optimizer import MemoryOptimizer


class ConversationSession:
    def __init__(self):
        self.session_id = None
        self.start_ts = time.time()
        self.title = "session"
        self.optimizer = MemoryOptimizer()

    def build_context(self, query: str, model: str = "gpt-4", limit: int | None = None) -> str:
        max_tokens = limit if limit is not None else 4096
        return self.optimizer.build_optimal_context(
            {"name": model, "max_tokens": max_tokens},
            current_task=query,
            conversation_id=self.session_id,
        )
