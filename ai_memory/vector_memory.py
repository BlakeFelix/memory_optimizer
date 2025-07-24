from __future__ import annotations

import os
import pickle
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import faiss


logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    id: str
    text: str
    timestamp: float


class VectorMemory:
    """Load and query FAISS vector memory with metadata."""

    def __init__(self, index_path: str | None = None) -> None:
        base = Path(os.getenv("LUNA_VECTOR_DIR", ".ai_memory"))
        default_index = base / "memory_store.index"
        self.index_path = Path(os.getenv("LUNA_VECTOR_INDEX", index_path or default_index))
        self.meta_path = self.index_path.with_suffix(".pkl")
        self.legacy_path = self.index_path.parent / f"{self.index_path.stem}.memories.pkl"
        base.mkdir(parents=True, exist_ok=True)
        self.index: faiss.Index | None = None
        self.memories: Dict[str, MemoryEntry] = {}
        self._ordered: List[MemoryEntry] = []

    def load(self) -> bool:
        """Load FAISS index and metadata."""
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
            except Exception as e:
                logger.warning("Failed to read index %s: %s", self.index_path, e)
                self.index = None
        else:
            logger.warning("Index file %s not found", self.index_path)
            self.index = None

        meta_obj = None
        for path in (self.legacy_path, self.meta_path):
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        meta_obj = pickle.load(f)
                    break
                except Exception as e:
                    logger.warning("Failed to read metadata %s: %s", path, e)
        if meta_obj is None:
            self.memories = {}
            self._ordered = []
            return False

        if isinstance(meta_obj, dict):
            mems: Dict[str, MemoryEntry] = {}
            ordered: List[MemoryEntry] = []
            for k, v in meta_obj.items():
                if isinstance(v, MemoryEntry):
                    entry = v
                elif isinstance(v, dict):
                    entry = MemoryEntry(id=v.get("id", k), text=v.get("text", ""), timestamp=float(v.get("timestamp", 0)))
                else:
                    continue
                mems[entry.id] = entry
                ordered.append(entry)
            self.memories = mems
            self._ordered = ordered
        elif isinstance(meta_obj, list):
            mems: Dict[str, MemoryEntry] = {}
            ordered: List[MemoryEntry] = []
            for item in meta_obj:
                if isinstance(item, dict):
                    mid = item.get("id") or str(len(ordered))
                    entry = MemoryEntry(id=mid, text=item.get("text", ""), timestamp=float(item.get("timestamp", 0)))
                    mems[mid] = entry
                    ordered.append(entry)
            self.memories = mems
            self._ordered = ordered
        else:
            self.memories = {}
            self._ordered = []

        if self.index and len(self._ordered) != self.index.ntotal:
            logger.warning(
                "Vector/metadata count mismatch: %d != %d",
                len(self._ordered),
                self.index.ntotal,
            )
        return True

    def search(self, query: str, top_k: int = 5) -> List[Tuple[MemoryEntry, float]]:
        if not self.index or not self._ordered:
            return []
        from .vector_embedder import _embed_text  # lazy to avoid circular import
        vec = _embed_text(query)
        D, I = self.index.search(vec, top_k)
        results: List[Tuple[MemoryEntry, float]] = []
        for dist, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self._ordered):
                results.append((self._ordered[idx], float(dist)))
        return results
