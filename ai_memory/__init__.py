"""ai_memory package exposing VectorMemory lazily."""

import os
import sys
import types

if os.environ.get("AIMEM_DISABLE_TORCH", "").lower() in {"1", "true"}:
    from .testing._stubs import FakeSentenceTransformer

    sys.modules.setdefault(
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

__all__ = ["VectorMemory"]


class _VectorMemoryProxy:
    def __call__(self, *args, **kwargs):
        from .vector_memory import VectorMemory
        return VectorMemory(*args, **kwargs)

    def __getattr__(self, attr):
        from .vector_memory import VectorMemory
        return getattr(VectorMemory, attr)


_proxy = _VectorMemoryProxy()


def __getattr__(name):
    if name == "VectorMemory":
        return _proxy
    raise AttributeError(name)

