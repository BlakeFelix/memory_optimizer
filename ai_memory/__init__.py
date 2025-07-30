"""
ai_memory package bootstrap.

* Exposes ``VectorMemory`` lazily so plain ``import ai_memory`` never pulls in
  FAISS / NumPy / Torch unless you actually touch vector search.

* If the environment variable **AIMEM_DISABLE_TORCH** is set to ``1`` or
  ``true`` (case‑insensitive), we monkey‑patch a lightweight stub so CLI
  subprocesses and CI runners can operate without installing the heavy
  ``torch`` + ``sentence‑transformers`` stack.
"""

from __future__ import annotations

import os
import sys
import types

# --------------------------------------------------------------------------- #
#  Optional dependency guard – skip Torch/Sentence‑Transformers if requested  #
# --------------------------------------------------------------------------- #

if os.getenv("AIMEM_DISABLE_TORCH", "").lower() in {"1", "true"}:
    from .testing._stubs import FakeSentenceTransformer

    sys.modules.setdefault(
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

# --------------------------------------------------------------------------- #
#  Lazy exposure of VectorMemory                                              #
# --------------------------------------------------------------------------- #

__all__ = ["VectorMemory"]


class _VectorMemoryProxy:
    """Indirection wrapper so we import heavy libs only on first use."""

    def __call__(self, *args, **kwargs):
        from .vector_memory import VectorMemory

        return VectorMemory(*args, **kwargs)

    def __getattr__(self, attr):
        from .vector_memory import VectorMemory

        return getattr(VectorMemory, attr)


_proxy = _VectorMemoryProxy()


def __getattr__(name: str):
    if name == "VectorMemory":
        return _proxy
    raise AttributeError(name)
