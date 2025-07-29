import json
import pickle
import subprocess
from pathlib import Path
import faiss
import pytest

pytestmark = pytest.mark.slow


def test_vectorize_json_auto(tmp_path):
    data = {
        "conversations": [
            {
                "messages": [
                    {"content": {"parts": ["hello"]}},
                    {"content": {"parts": ["world"]}},
                    {"content": {"parts": ["goodbye"]}},
                ]
            }
        ]
    }
    json_file = tmp_path / "conv.json"
    json_file.write_text(json.dumps(data))
    index = tmp_path / "vec.index"

    subprocess.run(
        [
            "python",
            "-m",
            "ai_memory.cli",
            "vectorize",
            str(json_file),
            "--vector-index",
            str(index),
            "--json-extract",
            "auto",
        ],
        check=True,
    )

    assert index.exists() and index.stat().st_size > 5 * 1024
    meta = index.with_suffix(".pkl")
    assert meta.exists()
    meta_data = pickle.load(open(meta, "rb"))
    idx = faiss.read_index(str(index))
    assert len(meta_data) == idx.ntotal == 3
