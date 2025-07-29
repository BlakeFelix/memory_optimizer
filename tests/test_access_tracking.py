import sqlite3
from ai_memory.memory_db import _ensure_schema
from ai_memory.memory_store import MemoryStore
from ai_memory.context_builder import ContextBuilder
from ai_memory.relevance_engine import RelevanceEngine


def test_access_increment_on_build():
    conn = sqlite3.connect(":memory:")
    _ensure_schema(conn)
    store = MemoryStore(conn)

    mem_id = store.add("hello world", importance=1.0)

    builder = ContextBuilder(store)
    engine = RelevanceEngine()
    scored = engine.score_all(store.get_all(), task="hello", conversation_id=None)
    builder.build_layers(scored, token_budget=100)

    new_mem = store.get_all()[mem_id]
    assert new_mem.access_count > 0
