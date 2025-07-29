import os
import subprocess


def _run(cmd, env=None):
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_cli_vacuum(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_MEMORY_ROOT", str(tmp_path / "ai_memory"))
    import importlib
    import ai_memory.memory_db as db
    import ai_memory.memory_store as ms
    importlib.reload(db)
    importlib.reload(ms)
    env = os.environ.copy()
    MemoryStore = ms.MemoryStore
    store = MemoryStore()
    for i in range(1100):
        store.add(f"m{i}")
    store.conn.close()

    out = _run("python -m ai_memory.cli vacuum", env=env)
    assert "Memory vacuum" in out

    store = MemoryStore()
    assert len(store.get_all()) <= 1000
    store.conn.close()

