import json
import os
import pickle
import subprocess
import zipfile
from pathlib import Path
import faiss
import platform
import pytest

pytestmark = pytest.mark.skipif(
    platform.release() == "6.14.0-27-generic",
    reason="Kernel 6.14.0-27 panics with subprocess",
)


def test_ingest_zip_with_vectors(tmp_path):
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.mkdir()
    dest.mkdir()

    data = {
        "conversations": [
            {
                "messages": [
                    {"content": {"parts": ["hello"]}},
                    {"content": {"parts": ["world"]}},
                    {"content": {"parts": ["bye"]}},
                ]
            }
        ]
    }
    zip_path = src / "logs.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("chatlog.json", json.dumps(data))

    index = tmp_path / "mem.index"

    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    subprocess.run(
        [
            "python",
            "-m",
            "ai_memory.cli",
            "ingest-zip",
            "--src",
            str(src),
            "--dest",
            str(dest),
            "--index",
            str(index),
        ],
        check=True,
        env=env,
        timeout=5,
    )

    meta = index.with_suffix(".pkl")
    assert index.exists() and meta.exists()
    idx = faiss.read_index(str(index))
    meta_data = pickle.load(open(meta, "rb"))
    assert idx.ntotal == len(meta_data) == 3
