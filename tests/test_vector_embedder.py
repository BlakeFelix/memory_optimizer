import os
import sys
import types
import pytest
from ai_memory.testing._stubs import FakeSentenceTransformer
from ai_memory.vector_embedder import embed_file, recall


@pytest.fixture(autouse=True)
def _stub_transformer(monkeypatch):
    fake_mod = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    monkeypatch.setattr(embed_file.__module__ + ".SentenceTransformer", FakeSentenceTransformer, raising=False)


def test_self_heal(tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("hello world")
    idx = tmp_path / "vec.faiss"

    embed_file(str(doc), str(idx), "dummy", factory="Flat")
    assert recall(str(doc), str(idx)) > 0.99

    # corrupt the index
    idx.write_bytes(os.urandom(64))

    embed_file(str(doc), str(idx), "dummy", factory="Flat")
    assert recall(str(doc), str(idx)) > 0.99

def test_self_heal_new_factory(tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("hello again")
    idx = tmp_path / "vec.faiss"
    embed_file(str(doc), str(idx), "dummy", factory="Flat")
    assert recall(str(doc), str(idx)) > 0.99

    idx.write_bytes(os.urandom(128))

    embed_file(str(doc), str(idx), "dummy", factory="Flat")
    score = recall(str(doc), str(idx))
    assert score > 0.99

