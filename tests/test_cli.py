import subprocess, json, os, sys, tempfile, time


def _run(cmd, env=None):
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_add_and_list():
    _run("python -m ai_memory.cli add 'pytest memory' -i 4.2 -c test_session")
    out = _run("python -m ai_memory.cli list -n 1 -c test_session")
    assert 'pytest memory' in out


def test_context():
    ctx = _run("python -m ai_memory.cli context 'pytest memory' --conversation-id test_session")
    assert 'pytest memory' in ctx

def test_export(tmp_path):
    out = tmp_path / "mem.json"
    _run(f"python -m ai_memory.cli export -o {out}")
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert any('pytest memory' in m['content'] for m in data)


def test_list_by_entity():
    _run("python -m ai_memory.cli add 'contact me at foo@example.com' -c entity_sess")
    out = _run("python -m ai_memory.cli list --entity foo@example.com -n 1")
    assert 'foo@example.com' in out


def test_import_command(tmp_path):
    json_file = tmp_path / "conv.json"
    data = {
        "user": "tester",
        "conversations": [
            {
                "id": "c1",
                "started_at": "2025-01-01T00:00:00Z",
                "messages": [
                    {"id": "m1", "sender": "user", "content": "hi", "timestamp": "2025-01-01T00:00:01Z"},
                    {"id": "m2", "sender": "assistant", "content": "hello", "timestamp": "2025-01-01T00:00:02Z"},
                ],
            }
        ],
    }
    json_file.write_text(json.dumps(data))
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    _run(f"python -m ai_memory.cli import {json_file}", env=env)
    db_path = tmp_path / "ai_memory" / "ai_memory.db"
    assert db_path.exists()
    import sqlite3

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM messages")
    count = cur.fetchone()[0]
    conn.close()
    assert count == 2


def test_console_script_import(tmp_path):
    json_file = tmp_path / "conv.json"
    data = {
        "user": "tester",
        "conversations": [
            {
                "id": "c1",
                "started_at": "2025-01-01T00:00:00Z",
                "messages": [
                    {"id": "m1", "sender": "user", "content": "hi", "timestamp": "2025-01-01T00:00:01Z"},
                    {"id": "m2", "sender": "assistant", "content": "hello", "timestamp": "2025-01-01T00:00:02Z"},
                ],
            }
        ],
    }
    json_file.write_text(json.dumps(data))
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    _run("pip install -e .", env=env)
    _run(f"aimem import {json_file}", env=env)
    db_path = tmp_path / "ai_memory" / "ai_memory.db"
    assert db_path.exists()
    import sqlite3

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM messages")
    count = cur.fetchone()[0]
    conn.close()
    assert count == 2
