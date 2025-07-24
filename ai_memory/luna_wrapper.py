from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime

from .vector_memory import VectorMemory


def main() -> int:
    parser = argparse.ArgumentParser(description="Query LUNA vector memory")
    parser.add_argument("query", help="Query text")
    parser.add_argument("--top", "-k", type=int, default=5, help="Number of results")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug or os.getenv("LUNA_DEBUG") else logging.INFO)
    logger = logging.getLogger(__name__)

    logger.debug("Current dir: %s", os.getcwd())
    logger.debug("LUNA_VECTOR_DIR: %s", os.getenv("LUNA_VECTOR_DIR"))
    logger.debug("LUNA_VECTOR_INDEX: %s", os.getenv("LUNA_VECTOR_INDEX"))

    if not os.path.exists(".ai_memory"):
        default_home = os.path.expanduser("~/memory_optimizer")
        if os.path.exists(os.path.join(default_home, ".ai_memory")):
            os.chdir(default_home)
            logger.debug("Changed to: %s", os.getcwd())

    vm = VectorMemory()
    if not vm.load():
        print("No vector memory found")
        return 1
    logger.debug("VectorMemory loaded %d memories from %s", len(vm.memories), vm.index_path)

    results = vm.search(args.query, args.top)
    if not results:
        print("No results")
        return 0
    for entry, score in results:
        ts = datetime.fromtimestamp(entry.timestamp).isoformat()
        snippet = entry.text.replace("\n", " ")
        summary = snippet if len(snippet) <= 120 else snippet[:117] + "..."
        print(f"{score:.3f}\t{ts}\t{summary}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
