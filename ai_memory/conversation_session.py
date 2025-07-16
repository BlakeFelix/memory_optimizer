from __future__ import annotations
import uuid, time
from ai_memory.memory_optimizer import MemoryOptimizer

class ConversationSession:
    def __init__(self, title: str | None = None):
        self.session_id = str(uuid.uuid4())
        self.title = title or "untitled"
        self.start_ts = time.time()
        self.optimizer = MemoryOptimizer()

    def build_context(self, user_prompt: str, model="gpt-4", limit=8000):
        return self.optimizer.build_optimal_context(
            {"name": model, "max_tokens": limit},
            current_task=user_prompt,
            conversation_id=self.session_id,
        )
