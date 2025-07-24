from __future__ import annotations

import argparse
from datetime import datetime

from .vector_memory import VectorMemory


def main() -> int:
    parser = argparse.ArgumentParser(description="Query LUNA vector memory")
    parser.add_argument("query", help="Query text")
    parser.add_argument("--top", "-k", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    vm = VectorMemory()
    if not vm.load():
        print("No vector memory found")
        return 1

    results = vm.search(args.query, args.top)
    if not results:
        print("No results")
        return 0
    for entry, score in results:
        ts = datetime.fromtimestamp(entry.timestamp).isoformat()
        snippet = entry.text.replace("\n", " ")[:120]
        print(f"{score:.3f}\t{ts}\t{snippet}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
