from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .memory_db import create_production_memory_system, extract_entities


_WORD = re.compile(r"[A-Za-z0-9]+")


class RelevanceEngine:
    def __init__(self, conn: Optional[sqlite3.Connection] = None) -> None:
        self.conn = conn or create_production_memory_system()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> List[str]:
        return _WORD.findall(text.lower())

    def _tfidf_vectors(self, docs: List[str]):
        tokens_list = [self._tokenize(t) for t in docs]
        df = Counter()
        for tokens in tokens_list:
            df.update(set(tokens))
        n = len(docs)
        idf = {w: math.log((1 + n) / (1 + df[w])) + 1 for w in df}
        vecs = []
        for tokens in tokens_list:
            tf = Counter(tokens)
            vec = {w: tf[w] * idf[w] for w in tf}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            vecs.append({w: v / norm for w, v in vec.items()})
        return vecs

    def _cosine(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        return sum(a[w] * b[w] for w in common)

    # ------------------------------------------------------------------
    # core API
    # ------------------------------------------------------------------
    def _fetch(self, conv_id: Optional[str] = None) -> List[Dict[str, str]]:
        cur = self.conn.cursor()
        if conv_id:
            cur.execute(
                "SELECT mem_id, conv_id, msg_id, content, importance, token_estimate, created_at FROM memory_fragments WHERE conv_id=?",
                (conv_id,),
            )
        else:
            cur.execute(
                "SELECT mem_id, conv_id, msg_id, content, importance, token_estimate, created_at FROM memory_fragments"
            )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def score_all(
        self,
        current_task: Optional[str] = None,
        conv_id: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        mems = self._fetch(conv_id)
        docs = [m["content"] for m in mems]
        if current_task:
            docs.append(current_task)
        vecs = self._tfidf_vectors(docs)
        task_vec = vecs[-1] if current_task else None
        now = datetime.now(tz=timezone.utc)
        q_entities = (
            {val for _, val in extract_entities(current_task)} if current_task else set()
        )
        results = []
        for idx, m in enumerate(mems):
            vec = vecs[idx]
            sim = self._cosine(vec, task_vec) if task_vec else 0.0
            age_hours = (
                now - datetime.fromisoformat(m["created_at"])
            ).total_seconds() / 3600.0
            recency = math.exp(-age_hours / 168)
            overlap = 0
            if q_entities:
                ents = {val for _, val in extract_entities(m["content"])}
                overlap = len(q_entities & ents)
            score = sim * 50 + recency * 10 + overlap * 3 + m["importance"] * 5
            results.append(
                {
                    "mem_id": m["mem_id"],
                    "content": m["content"],
                    "score": score,
                    "token_cost": m["token_estimate"],
                }
            )
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
