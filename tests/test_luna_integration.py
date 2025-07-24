import os
import json
import subprocess
from ai_memory.vector_memory import VectorMemory

def test_luna_integration(tmp_path):
    data = {
        "conversations": [
            {"messages": [{"content": {"parts": ["hello world"]}}]}
        ]
    }
    json_file = tmp_path / "conv.json"
    json_file.write_text(json.dumps(data))
    index = tmp_path / "memory_store.index"

    env = os.environ.copy()
    env["LUNA_VECTOR_DIR"] = str(tmp_path)
    env["LUNA_VECTOR_INDEX"] = str(index)
    os.environ.update(env)

    subprocess.run([
        "python",
        "-m",
        "ai_memory.cli",
        "vectorize",
        str(json_file),
        "--vector-index",
        str(index),
        "--json-extract",
        "messages",
    ], check=True, env=env)

    meta_file = index.with_suffix(".pkl")
    legacy_file = index.parent / f"{index.stem}.memories.pkl"
    if legacy_file.exists():
        legacy_file.unlink()

    subprocess.run(
        ["python", "-m", "ai_memory.cli", "convert-metadata", str(meta_file)],
        check=True,
        env=env,
    )

    result = subprocess.run(
        ["python", "-m", "ai_memory.luna_wrapper", "hello"],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
        check=True,
    )
    vm = VectorMemory()
    vm.load()
    assert len(vm.memories) > 0
    assert "hello" in result.stdout.lower()
