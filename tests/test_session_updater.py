import importlib
import os
import ai_memory.conversation_session as cs


def test_session_updater_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_MEMORY_ROOT", str(tmp_path / "ai_memory"))
    importlib.reload(cs)
    session = cs.ConversationSession()
    session.session_id = "sess"
    for i in range(1100):
        session.add_exchange(f"u{i}", f"a{i}")
    all_mem = session.optimizer.memory_store.get_all()
    assert len(all_mem) <= 1000
    assert any(m.type == "summary" for m in all_mem.values())


