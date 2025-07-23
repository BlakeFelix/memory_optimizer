from __future__ import annotations

from pathlib import Path
import logging
import os

import numpy as np
import faiss

logger = logging.getLogger(__name__)


_DIMS = 1


def _embed(file: str) -> np.ndarray:
    """Return a trivial embedding for the given file."""
    with open(file, "rb") as f:
        length = len(f.read())
    vec = np.array([[float(length)]], dtype="float32")
    faiss.normalize_L2(vec)
    return vec


def _create_index(factory: str | None) -> faiss.Index:
    if factory:
        return faiss.index_factory(_DIMS, factory)
    return faiss.IndexFlatIP(_DIMS)


def _load_index(path: Path, factory: str | None) -> faiss.Index:
    if not path.exists():
        return _create_index(factory)
    try:
        return faiss.read_index(str(path))
    except Exception as e:  # faiss may raise IOError or RuntimeError
        if "not recognized" in str(e):
            logger.warning("Index read error: %s", e)
            try:
                os.rename(path, path.with_suffix(path.suffix + ".corrupt"))
            except OSError:
                pass
            # rebuild with a simple IndexFlatIP for compatibility
            return faiss.IndexFlatIP(_DIMS)
        raise


def embed_file(file: str, index_path: str, model: str, factory: str | None = None) -> None:
    """Embed a file into a FAISS index."""
    index_file = Path(index_path)
    index_file.parent.mkdir(parents=True, exist_ok=True)
    index = _load_index(index_file, factory)
    vec = _embed(file)
    if not index.is_trained and hasattr(index, "train"):
        index.train(vec)
    index.add(vec)
    faiss.write_index(index, str(index_file))


def recall(file: str, index_path: str) -> float:
    """Return similarity score for the file against the index."""
    index = faiss.read_index(str(index_path))
    vec = _embed(file)
    D, _ = index.search(vec, 1)
    score = float(D[0][0])
    if index.metric_type == faiss.METRIC_L2:
        score = 1.0 - score
    return score
