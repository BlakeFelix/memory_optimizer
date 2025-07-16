from typing import List

from .database import MemoryDatabase


def import_files(db: MemoryDatabase, paths: List[str]) -> None:
    """Batch import a list of aimemory JSON files."""
    for p in paths:
        try:
            db.import_json(p)
        except Exception as exc:  # pragma: no cover - import failure
            print(f"Failed to import {p}: {exc}")

