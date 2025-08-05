import json
import os
import subprocess
import zipfile
import hashlib
import sqlite3
from pathlib import Path
import platform
import pytest

pytestmark = pytest.mark.skipif(
    platform.release() == "6.14.0-27-generic",
    reason="Kernel 6.14.0-27 panics with subprocess",
)


def _run(cmd, env=None):
    result = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def test_ingest_zip(tmp_path):
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.mkdir()
    dest.mkdir()
    data = {
        "user": "tester",
        "conversations": [
            {
                "id": "c1",
                "started_at": "2025-01-01T00:00:00Z",
                "messages": [
                    {
                        "id": "m1",
                        "sender": "user",
                        "content": "contact foo@example.com",
                        "timestamp": "2025-01-01T00:00:01Z",
                        "entities": [{"value": "foo@example.com"}],
                    }
                ],
            }
        ],
    }
    zip_path = src / "logs.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("chat_memory.json", json.dumps(data))

    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    _run(f"python -m ai_memory.cli ingest-zip --src {src} --dest {dest}", env=env)

    db_path = Path(env["HOME"]) / "ai_memory" / "ai_memory.db"
    assert db_path.exists()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    message_id = md5("m1")
    entity_id = md5("foo@example.com")
    mem_id = md5(message_id + entity_id)
    cur.execute("SELECT fragment_id FROM memory_fragments WHERE fragment_id=?", (mem_id,))
    row = cur.fetchone()
    conn.close()
    assert row is not None
