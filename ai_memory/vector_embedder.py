from __future__ import annotations

from pathlib import Path


def embed_file(file: str, index_path: str, model: str) -> None:
    """Trivial stand-in for vector embedding logic."""
    index = Path(index_path)
    index.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "rb") as f:
        data = f.read()
    # In lieu of a real embedding we just record the file length
    with open(index, "a", encoding="utf-8") as idx:
        idx.write(f"{Path(file).name}:{len(data)}\n")
