from typing import Dict, List


class ContextBuilder:
    def build_layers(
        self, scored_memories: List[Dict], token_budget: int
    ) -> str:
        memories_by_value = sorted(
            scored_memories,
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
            if tokens_used + mem["token_cost"] <= essential_budget:
                context_layers["essential"].append(mem)
                tokens_used += mem["token_cost"]

        relevant_budget = token_budget * 0.6
        for mem in memories_by_value:
            if mem not in context_layers["essential"] and mem["score"] > 15:
                if tokens_used + mem["token_cost"] <= relevant_budget:
                    context_layers["relevant"].append(mem)
                    tokens_used += mem["token_cost"]

        for mem in memories_by_value:
            if mem not in context_layers["essential"] + context_layers["relevant"]:
                if tokens_used + mem["token_cost"] <= token_budget * 0.95:
                    context_layers["supplemental"].append(mem)
                    tokens_used += mem["token_cost"]

        return self._format_context(context_layers, tokens_used)

    def _format_context(self, layers: Dict[str, List[Dict]], tokens_used: int) -> str:
        formatted = [f"TOTAL TOKENS USED: {tokens_used}"]
        for layer in ["essential", "relevant", "supplemental"]:
            formatted.append(f"=== {layer.upper()} ===")
            for mem in layers[layer]:
                formatted.append(f"[{mem['mem_id']}] {mem['content']}")
        return "\n".join(formatted)
