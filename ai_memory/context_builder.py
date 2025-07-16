"""
Layered context assembly.

* No longer relies on a 'type' field that isn't present.
* Guarantees a non\-empty context so long as there is at least one memory.
"""

from __future__ import annotations
from typing import List, Dict


class ContextBuilder:
    # ------------------------------------------------------------------ #
    # public                                                             #
    # ------------------------------------------------------------------ #
    def build_layers(self, scored_memories: List[Dict], token_budget: int) -> str:
        """Return a context string honouring the token_budget."""
        memories = sorted(
            scored_memories,
            key=lambda m: m["score"] / max(m["token_cost"], 1),
            reverse=True,
        )

        essential: List[Dict] = []
        relevant: List[Dict] = []
        supplemental: List[Dict] = []

        tokens_used = 0
        hard_cap = int(token_budget * 0.95)

        # -------------------------------------------------------------- #
        # 1) top\-scored memories marked *high importance* (if any)       #
        #    — treat as 'essential' up to 20\% budget                    #
        # -------------------------------------------------------------- #
        essential_budget = int(token_budget * 0.20)
        for mem in memories:
            if tokens_used + mem["token_cost"] > essential_budget:
                continue
            if mem["score"] >= 40:            # heuristic “very relevant”
                essential.append(mem)
                tokens_used += mem["token_cost"]

        # -------------------------------------------------------------- #
        # 2) next best memories until 80\% budget                        #
        # -------------------------------------------------------------- #
        relevant_budget = int(token_budget * 0.80)
        for mem in memories:
            if mem in essential:
                continue
            if tokens_used + mem["token_cost"] > relevant_budget:
                continue
            relevant.append(mem)
            tokens_used += mem["token_cost"]

        # -------------------------------------------------------------- #
        # 3) fill the remaining 15\% with whatever fits                  #
        # -------------------------------------------------------------- #
        for mem in memories:
            if mem in essential or mem in relevant:
                continue
            if tokens_used + mem["token_cost"] > hard_cap:
                continue
            supplemental.append(mem)
            tokens_used += mem["token_cost"]

        # -------------------------------------------------------------- #
        # 4) safeguard: if context still empty, drop *something*         #
        # -------------------------------------------------------------- #
        if not (essential or relevant or supplemental) and memories:
            first = memories[0]
            supplemental.append(first)

        return self._format(essential, relevant, supplemental)

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _format(
        essential: List[Dict],
        relevant: List[Dict],
        supplemental: List[Dict],
    ) -> str:
        ordered = (
            [m["content"] for m in essential]
            + [m["content"] for m in relevant]
            + [m["content"] for m in supplemental]
        )
        return "\n".join(ordered)
