import os
import sys
import types
import sqlite3
import pytest
from ai_memory.testing._stubs import FakeSentenceTransformer
from ai_memory.vector_embedder import embed_file
from ai_memory.memory_db import _ensure_schema
from ai_memory.memory_store import MemoryStore
from ai_memory.relevance_engine import RelevanceEngine

@pytest.fixture(autouse=True)
def _stub_transformer(monkeypatch):
    fake_mod = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    monkeypatch.setattr(embed_file.__module__ + ".SentenceTransformer", FakeSentenceTransformer, raising=False)


def test_vector_hit_scored(tmp_path, monkeypatch):
    # prepare vector index
    txt = tmp_path / "doc.txt"
    txt.write_text("vector memory test")
    index = tmp_path / "mem.index"
    embed_file(str(txt), str(index), "dummy", factory="Flat")

    monkeypatch.setenv("LUNA_VECTOR_DIR", str(tmp_path))
    monkeypatch.setenv("LUNA_VECTOR_INDEX", str(index))

    conn = sqlite3.connect(":memory:")
    _ensure_schema(conn)
    store = MemoryStore(conn)

    engine = RelevanceEngine()
    scores = engine.score_all(store.get_all(), "vector memory test", None)
    contents = [v["memory"].content for v in scores.values()]
    assert "vector memory test" in contents


def test_partial_match(tmp_path, monkeypatch):
    txt = tmp_path / "doc.txt"
    txt.write_text("alpha beta")
    index = tmp_path / "mem.index"
    embed_file(str(txt), str(index), "dummy", factory="Flat")

    monkeypatch.setenv("LUNA_VECTOR_DIR", str(tmp_path))
    monkeypatch.setenv("LUNA_VECTOR_INDEX", str(index))

    conn = sqlite3.connect(":memory:")
    _ensure_schema(conn)
    store = MemoryStore(conn)

    engine = RelevanceEngine()
    scores = engine.score_all(store.get_all(), "beta", None)
    contents = [v["memory"].content for v in scores.values()]
    assert "alpha beta" in contents
