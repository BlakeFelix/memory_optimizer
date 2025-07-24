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
        search_dirs = [Path("."), Path(os.getenv("LUNA_VECTOR_DIR", ".ai_memory")), Path(".ai_memory")]
        candidates = []
        if self.index_path.is_absolute():
            candidates.append(self.index_path)
        else:
            candidates.append(self.index_path)
            for d in search_dirs:
                candidates.append(d / self.index_path.name)

        index_file = None
        for cand in candidates:
            logger.debug("Looking for index file at %s", cand)
            if cand.exists():
                index_file = cand
                break
        if index_file:
            self.index_path = index_file
            try:
                self.index = faiss.read_index(str(index_file))
            except Exception as e:
                logger.warning("Failed to read index %s: %s", index_file, e)
                self.index = None
        else:
            logger.error("Index file not found. Tried: %s", ", ".join(str(c) for c in candidates))
            self.index = None

        meta_obj = None
        meta_candidates = [
            self.legacy_path,
            self.meta_path,
            self.index_path.with_suffix(".memories.pkl"),
            self.index_path.with_suffix(".pkl"),
        ]
        for mpath in meta_candidates:
            logger.debug("Looking for metadata file at %s", mpath)
            if mpath.exists():
                try:
                    with open(mpath, "rb") as f:
                        meta_obj = pickle.load(f)
                    path = mpath
                    break
                except Exception as e:
                    logger.warning("Failed to read metadata %s: %s", mpath, e)
        if meta_obj is not None:
            logger.info("Loaded metadata from %s", path)
        if meta_obj is None:
            logger.error("Metadata files not found")
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
                    ts = v.get("timestamp", 0)
                    entry = MemoryEntry(
                        id=v.get("id", k),
                        text=v.get("text", ""),
                        timestamp=ts.timestamp() if hasattr(ts, "timestamp") else float(ts),
                    )
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
                    ts = item.get("timestamp", 0)
                    entry = MemoryEntry(
                        id=mid,
                        text=item.get("text", ""),
                        timestamp=ts.timestamp() if hasattr(ts, "timestamp") else float(ts),
                    )
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
        logger.info("Loaded %d memories", len(self._ordered))
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
