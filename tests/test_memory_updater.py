import sqlite3
from ai_memory.memory_db import _ensure_schema
from ai_memory.memory_store import MemoryStore
from ai_memory.memory_updater import MemoryUpdater


def test_auto_summarisation():
    conn = sqlite3.connect(":memory:")
    _ensure_schema(conn)
    store = MemoryStore(conn)

    for i in range(1200):
        store.add(f"mem {i}", importance=0.1)

    updater = MemoryUpdater(store, max_size=1000)
    updater.post_conversation_update("user and assistant")

    all_mem = store.get_all()
    assert len(all_mem) == 1000
    assert any(m.type == "summary" for m in all_mem.values())

