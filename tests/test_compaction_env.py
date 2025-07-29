import sqlite3
from ai_memory.memory_db import _ensure_schema
from ai_memory.memory_store import MemoryStore
from ai_memory.memory_updater import MemoryUpdater
from ai_memory.token_counter import TokenCounter


def test_env_knobs(monkeypatch):
    monkeypatch.setenv("AIMEM_MAX_MEMORIES", "10")
    monkeypatch.setenv("AIMEM_SUMMARY_TOKENS", "10")
    monkeypatch.setenv("AIMEM_COMPRESS_BATCH", "2")
    conn = sqlite3.connect(":memory:")
    _ensure_schema(conn)
    store = MemoryStore(conn)
    for i in range(15):
        store.add(f"m{i}", importance=0.1)
    updater = MemoryUpdater(store)
    updater.post_conversation_update("x")
    assert len(store.get_all()) > 10
    for _ in range(5):
        updater.post_conversation_update("x")
        if len(store.get_all()) <= 10:
            break
    assert len(store.get_all()) <= 10
    summary = next(m for m in store.get_all().values() if m.type == "summary")
    tk = TokenCounter()
    assert tk.count(summary.content) <= 10
