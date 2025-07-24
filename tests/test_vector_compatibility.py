import os
import subprocess

from ai_memory.vector_memory import VectorMemory


def test_vector_compatibility(tmp_path):
    txt = tmp_path / "doc.txt"
    txt.write_text("hello world")
    index = tmp_path / "memory_store.index"

    env = os.environ.copy()
    env["LUNA_VECTOR_DIR"] = str(tmp_path)
    env["LUNA_VECTOR_INDEX"] = str(index)
    os.environ.update(env)

    subprocess.run(
        [
            "python",
            "-m",
            "ai_memory.cli",
            "vectorize",
            str(txt),
            "--vector-index",
            str(index),
            "--factory",
            "Flat",
        ],
        check=True,
        env=env,
    )

    vm = VectorMemory()
    vm.load()
    assert len(vm.memories) > 0
