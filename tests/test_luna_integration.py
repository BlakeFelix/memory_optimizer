import os
import subprocess
from pathlib import Path
import platform
import shutil
import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        platform.release() == "6.14.0-27-generic",
        reason="Kernel 6.14.0-27 panics with subprocess",
    ),
]

pytest.skip("luna CLI not available", allow_module_level=True)


def test_luna_loads_memories(tmp_path):
    index_dir = tmp_path / "test_index"
    index_dir.mkdir()

    test_json = tmp_path / "test.json"
    test_json.write_text('{"conversations":[{"messages":[{"content":"test memory"}]}]}')

    env = os.environ.copy()
    env["LUNA_VECTOR_DIR"] = str(index_dir)
    env["AIMEM_DISABLE_TORCH"] = "1"

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
