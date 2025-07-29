import os
import subprocess
import json
import pickle
import pytest

pytestmark = pytest.mark.slow

from ai_memory.vector_memory import VectorMemory


def test_vector_compatibility(tmp_path):
    txt = tmp_path / "doc.txt"
    txt.write_text("hello world")
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
        str(txt),
        "--vector-index",
        str(index),
        "--factory",
        "Flat",
    ], check=True, env=env)

    meta_file = index.with_suffix(".pkl")
    legacy_file = index.parent / f"{index.stem}.memories.pkl"
    assert meta_file.exists()
    assert legacy_file.exists()
    meta = pickle.load(open(meta_file, "rb"))
    legacy = pickle.load(open(legacy_file, "rb"))
    assert isinstance(meta, list)
    assert isinstance(legacy, dict)

    # vectorize a json conversation as well
    conv = {"conversations": [{"messages": [{"content": {"parts": ["hello"]}}]}]}
    json_file = tmp_path / "conv.json"
    json_file.write_text(json.dumps(conv))
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

    vm = VectorMemory()
    vm.load()
    assert len(vm.memories) > 0
    assert len(vm.memories) == vm.index.ntotal
