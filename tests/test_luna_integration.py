import os
import subprocess
from pathlib import Path
import pytest

pytestmark = pytest.mark.slow


def test_luna_loads_memories(tmp_path):
    index_dir = tmp_path / "test_index"
    index_dir.mkdir()

    test_json = tmp_path / "test.json"
    test_json.write_text('{"conversations":[{"messages":[{"content":"test memory"}]}]}')

    env = os.environ.copy()
    env["LUNA_VECTOR_DIR"] = str(index_dir)

    subprocess.run([
        "aimem", "vectorize", str(test_json),
        "--vector-index", str(index_dir / "memory_store.index"),
        "--json-extract", "messages"
    ], check=True, env=env)

    result = subprocess.run(
        ["luna", "test"],
        capture_output=True,
        text=True,
        env=env
    )

    assert "Loaded" in result.stderr
    assert "memories" in result.stderr
    assert "Loaded 0 memories" not in result.stderr
