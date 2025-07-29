from typing import Dict, List

from .memory import Memory
from .memory_store import MemoryStore


class ContextBuilder:
    def __init__(self, store: MemoryStore) -> None:
        self.memory_store = store
    def build_layers(
        self, scored_memories: Dict[str, Dict], token_budget: int
    ):
        memories_by_value = sorted(
            scored_memories.values(),
            key=lambda x: x["score"] / max(x["token_cost"], 1),
            reverse=True,
        )

        context_layers = {
            "essential": [],
            "relevant": [],
            "supplemental": [],
        }

        tokens_used = 0
        essential_budget = token_budget * 0.2
        for mem in memories_by_value:
            if mem["memory"].type in ["core_identity", "active_project_state"]:
                if tokens_used + mem["token_cost"] <= essential_budget:
                    context_layers["essential"].append(mem["memory"])
                    tokens_used += mem["token_cost"]

        relevant_budget = token_budget * 0.6
        for mem in memories_by_value:
            if (
                mem["score"] > 15
                and mem["memory"] not in context_layers["essential"]
            ):
                if tokens_used + mem["token_cost"] <= relevant_budget:
                    context_layers["relevant"].append(mem["memory"])
                    tokens_used += mem["token_cost"]

        for mem in memories_by_value:
            if (
                mem["memory"]
                not in context_layers["essential"] + context_layers["relevant"]
            ):
                if tokens_used + mem["token_cost"] <= token_budget * 0.95:
                    context_layers["supplemental"].append(mem["memory"])
                    tokens_used += mem["token_cost"]

        for m in (
            context_layers["essential"]
            + context_layers["relevant"]
            + context_layers["supplemental"]
        ):
            self.memory_store.update_access(m.memory_id)

        return self._format_context(context_layers, tokens_used)

    def _format_context(
        self, layers: Dict[str, List[Memory]], tokens_used: int
    ):
        formatted = []
        for layer in ["essential", "relevant", "supplemental"]:
            for mem in layers[layer]:
                formatted.append(mem.content)
        return "\n".join(formatted)
