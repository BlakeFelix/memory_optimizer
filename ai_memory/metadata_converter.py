from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import click

from .vector_memory import MemoryEntry


def convert_to_memory_entry_format(pkl_path: str, output_path: str) -> int:
    """Convert list-based metadata to dict-of-MemoryEntry format."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        raise ValueError("Input metadata must be a list of dictionaries")

    result: Dict[str, dict] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        mem_id = item.get("id") or str(len(result))
        ts = float(item.get("timestamp", 0))
        result[mem_id] = {
            "text": item.get("text", ""),
            "embedding": None,
            "metadata": {},
            "timestamp": ts,
            "access_count": 1,
            "last_accessed": ts,
            "importance": 1.0,
            "compressed": False,
        }

    with open(output_path, "wb") as f:
        pickle.dump(result, f, protocol=4)

    return len(result)


@click.command(name="convert-metadata")
@click.argument("pkl_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output .memories.pkl path")
def cli(pkl_path: str, output: str | None) -> None:
    """Convert aimem metadata to VectorMemory format."""
    out = output or str(Path(pkl_path).with_suffix(".memories.pkl"))
    count = convert_to_memory_entry_format(pkl_path, out)
    click.echo(f"\u2713 Converted {count} entries to {out}")


if __name__ == "__main__":  # pragma: no cover
    cli()
