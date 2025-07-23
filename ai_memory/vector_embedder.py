from __future__ import annotations

from pathlib import Path
import logging
import os
import json

import numpy as np
import faiss

logger = logging.getLogger(__name__)


_DIMS = 512


def _embed_text(text: str) -> np.ndarray:
    """Return an embedding for the given text."""
    length = float(len(text.encode("utf-8")))
    vec = np.full((_DIMS,), length, dtype="float32")
    vec = vec.reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec


def _embed(file: str) -> np.ndarray:
    """Return a trivial embedding for the given file."""
    with open(file, "rb") as f:
        length = len(f.read())
    vec = np.full((_DIMS,), float(length), dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec


def _create_index(factory: str | None) -> faiss.Index:
    if factory:
        return faiss.index_factory(_DIMS, factory)
    return faiss.IndexFlatIP(_DIMS)


def _extract_strings(obj, max_size=2048):
    strings: list[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            strings.extend(_extract_strings(v, max_size))
    elif isinstance(obj, list):
        for item in obj:
            strings.extend(_extract_strings(item, max_size))
    elif isinstance(obj, str):
        if len(obj.encode("utf-8")) < max_size:
            strings.append(obj)
    return strings


def _extract_messages(obj) -> list[str]:
    messages = []
    if not isinstance(obj, dict):
        return messages
    convs = obj.get("conversations")
    if isinstance(convs, list):
        for conv in convs:
            msgs = conv.get("messages") if isinstance(conv, dict) else None
            if isinstance(msgs, list):
                for m in msgs:
                    if isinstance(m, dict):
                        content = m.get("content")
                        if isinstance(content, str) and len(content.encode("utf-8")) < 2048:
                            messages.append(content)
    return messages


def _json_strings(path: str, mode: str) -> list[str] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("JSON parse error for %s: %s", path, e)
        return None
    if not isinstance(data, (dict, list)):
        return None
    if mode == "messages":
        texts = _extract_messages(data)
    elif mode == "all":
        texts = _extract_strings(data)
    elif mode == "auto":
        texts = _extract_messages(data)
        if not texts:
            texts = _extract_strings(data)
    else:
        return None
    return texts


def _json_to_text(path: str, mode: str) -> str | None:
    texts = _json_strings(path, mode)
    if not texts:
        return None
    return "\n\n".join(texts)


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


def embed_file(
    file: str,
    index_path: str,
    model: str,
    factory: str | None = None,
    json_extract: str = "auto",
) -> None:
    """Embed a file into a FAISS index."""
    index_file = Path(index_path)
    index_file.parent.mkdir(parents=True, exist_ok=True)
    index = _load_index(index_file, factory)
    vecs = None
    if file.endswith(".json") and json_extract != "none":
        texts = _json_strings(file, json_extract)
        if texts:
            vecs = np.vstack([_embed_text(t) for t in texts])
    if vecs is None:
        vecs = _embed(file)
    if not index.is_trained and hasattr(index, "train"):
        index.train(vecs)
    index.add(vecs)
    faiss.write_index(index, str(index_file))


def recall(file: str, index_path: str, json_extract: str = "auto") -> float:
    """Return similarity score for the file against the index."""
    index = faiss.read_index(str(index_path))
    vec = None
    if file.endswith(".json") and json_extract != "none":
        texts = _json_strings(file, json_extract)
        if texts:
            text = "\n\n".join(texts)
            vec = _embed_text(text)
    if vec is None:
        vec = _embed(file)
    D, _ = index.search(vec, 1)
    score = float(D[0][0])
    if index.metric_type == faiss.METRIC_L2:
        score = 1.0 - score
    return score
