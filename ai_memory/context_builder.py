from typing import List, Dict

class ContextBuilder:
    def build_layers(self, scored_memories: List[Dict], token_budget: int) -> str:
        memories_by_value = sorted(
            scored_memories,
            key=lambda x: x["score"] / max(x["token_cost"], 1),
            reverse=True,
        )

        layers = {"essential": [], "relevant": [], "supplemental": []}
        tokens_used = 0

        # 1) essential (20 %)
        for mem in memories_by_value:
            if mem["token_cost"] + tokens_used > token_budget * 0.2:
                continue
            if mem.get("type") in ("core_identity", "active_project_state"):
                layers["essential"].append(mem)
                tokens_used += mem["token_cost"]

        # 2) relevant (60 %)
        for mem in memories_by_value:
            if mem in layers["essential"]:
                continue
            if mem["score"] > 15 and mem["token_cost"] + tokens_used <= token_budget * 0.8:
                layers["relevant"].append(mem)
                tokens_used += mem["token_cost"]

        # 3) supplemental (fill → 95 %)
        for mem in memories_by_value:
            if mem in layers["essential"] or mem in layers["relevant"]:
                continue
            if mem["token_cost"] + tokens_used <= token_budget * 0.95:
                layers["supplemental"].append(mem)
                tokens_used += mem["token_cost"]

        return self._format(layers)

    @staticmethod
    def _format(layers: Dict[str, List[Dict]]) -> str:
        ordered = (
            [m["content"] for m in layers["essential"]]
            + [m["content"] for m in layers["relevant"]]
            + [m["content"] for m in layers["supplemental"]]
        )
        return "\n".join(ordered)
