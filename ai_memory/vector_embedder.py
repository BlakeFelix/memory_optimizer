from __future__ import annotations

from pathlib import Path
import logging
import os
import json
import pickle
import time
from uuid import uuid4
from typing import Dict

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

import numpy as np
import faiss

logger = logging.getLogger(__name__)


_model = None
_model_name = "BAAI/bge-large-en-v1.5"

_DIMS = 1024  # BAAI/bge-large-en-v1.5 uses 1024 dimensions


def _get_model():
    global _model
    if _model is None:
        device = "cpu"
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "" and torch is not None:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                pass
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")
        kwargs = {}
        if os.getenv("HF_HUB_OFFLINE") or os.getenv("LUNA_OFFLINE_TEST"):
            kwargs["local_files_only"] = True
        _model = SentenceTransformer(_model_name, device=device, **kwargs)
    return _model


def _embed_text(text: str) -> np.ndarray:
    """Return a real embedding for the given text using sentence-transformers."""
    try:
        model = _get_model()
        embedding = model.encode([text], convert_to_numpy=True)
        return embedding.astype("float32")
    except Exception as e:
        logger.warning(f"Failed to create real embedding, falling back to dummy: {e}")
        length = float(len(text.encode("utf-8")))
        vec = np.full((_DIMS,), length, dtype="float32")
        vec = vec.reshape(1, -1)
        faiss.normalize_L2(vec)
        return vec


def _embed(file: str) -> np.ndarray:
    """Return an embedding for the given file."""
    try:
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return _embed_text(text)
    except Exception:
        # Fallback for binary files or errors
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


def _iter_messages(obj):
    if not isinstance(obj, dict):
        return
    convs = obj.get("conversations")
    if isinstance(convs, list):
        for conv in convs:
            msgs = conv.get("messages") if isinstance(conv, dict) else None
            if isinstance(msgs, list):
                for m in msgs:
                    if isinstance(m, dict):
                        content = m.get("content")
                        if isinstance(content, dict):
                            parts = content.get("parts")
                            if isinstance(parts, list):
                                for p in parts:
                                    if (
                                        isinstance(p, str)
                                        and len(p.encode("utf-8")) <= 2048
                                    ):
                                        yield p
                        elif (
                            isinstance(content, str)
                            and len(content.encode("utf-8")) <= 2048
                        ):
                            yield content


def _iter_json_strings(path: str, mode: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("JSON parse error for %s: %s", path, e)
        return []
    if not isinstance(data, (dict, list)):
        return []
    if mode == "messages":
        return list(_iter_messages(data))
    if mode == "all":
        return _extract_strings(data)
    if mode == "auto":
        texts = list(_iter_messages(data))
        return texts or _extract_strings(data)
    return []


def _json_to_text(path: str, mode: str) -> str | None:
    texts = _iter_json_strings(path, mode)
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
    *,
    no_meta: bool = False,
    verbose: bool = False,
) -> int:
    """Embed a file into a FAISS index."""
    start = time.time()
    index_file = Path(index_path)
    index_file.parent.mkdir(parents=True, exist_ok=True)
    index = _load_index(index_file, factory)
    chunks: list[str] = []
    vecs = None
    if file.endswith(".json") and json_extract != "none":
        chunks = list(_iter_json_strings(file, json_extract))
        if chunks:
            vecs = np.vstack([_embed_text(t) for t in chunks])
    if vecs is None:
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        chunks = [text]
        vecs = _embed(file)
    if not index.is_trained and hasattr(index, "train"):
        index.train(vecs)
    index.add(vecs)
    faiss.write_index(index, str(index_file))

    if verbose:
        logger.info("extracted %d chunks", len(chunks))
        logger.info("added %d vectors", vecs.shape[0])

    if not no_meta:
        meta_path = index_file.with_suffix(".pkl")
        legacy_path = index_file.parent / f"{index_file.stem}.memories.pkl"
        meta: list[dict] = []
        if meta_path.exists():
            try:
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
            except Exception:
                meta = []

        for chunk in chunks:
            meta.append({"id": uuid4().hex, "text": chunk, "timestamp": time.time()})

        if len(meta) != index.ntotal:
            logger.warning("metadata count mismatch: %d != %d", len(meta), index.ntotal)
            if len(meta) > index.ntotal:
                meta = meta[: index.ntotal]
            else:
                for _ in range(index.ntotal - len(meta)):
                    meta.append(
                        {"id": uuid4().hex, "text": "", "timestamp": time.time()}
                    )

        with open(meta_path, "wb") as f:
            pickle.dump(meta, f, protocol=4)

        # VectorMemory dictionary format
        legacy: Dict[str, dict] = {}
        for m in meta:
            ts = float(m["timestamp"])
            legacy[m["id"]] = {
                "text": m.get("text", ""),
                "embedding": None,
                "metadata": {},
                "timestamp": ts,
                "access_count": 1,
                "last_accessed": ts,
                "importance": 1.0,
                "compressed": False,
            }
        with open(legacy_path, "wb") as f:
            pickle.dump(legacy, f, protocol=4)

    if verbose:
        logger.info("took %.2fs", time.time() - start)

    status = 0
    if not no_meta:
        status = 0 if len(meta) == index.ntotal else 1
    return status


def recall(file: str, index_path: str, json_extract: str = "auto") -> float:
    """Return similarity score for the file against the index."""
    index = faiss.read_index(str(index_path))
    vec = None
    if file.endswith(".json") and json_extract != "none":
        texts = _iter_json_strings(file, json_extract)
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
