import json
import faiss
from ai_memory.vector_embedder import embed_file


def test_vectorize_json_messages(tmp_path):
    data = {
        "conversations": [
            {"messages": [
                {"content": "hello"},
                {"content": "world"},
                {"content": "goodbye"},
            ]}
        ]
    }
    json_file = tmp_path / "conversations.json"
    json_file.write_text(json.dumps(data))
    index = tmp_path / "vec.faiss"

    embed_file(str(json_file), str(index), "dummy", factory="Flat", json_extract="messages")

    idx = faiss.read_index(str(index))
    assert idx.ntotal == 3
    assert index.stat().st_size > 2048
