
"""Wrapper for querying vector memories similar to the original LUNA script."""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Query LUNA vector memory")
    parser.add_argument("query", help="Query text")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.debug(f"Current dir: {os.getcwd()}")
    logger.debug(f"LUNA_VECTOR_DIR: {os.getenv('LUNA_VECTOR_DIR')}")

    # Force CPU if CUDA is disabled via environment
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
        logger.info("Running in CPU mode (CUDA disabled)")
        device = 'cpu'
    else:
        try:
            import torch  # type: ignore
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            device = 'cpu'

    if not os.path.exists(".ai_memory"):
        for candidate in ["~/memory_optimizer", "."]:
            expanded = os.path.expanduser(candidate)
            if os.path.exists(os.path.join(expanded, ".ai_memory")):
                os.chdir(expanded)
                logger.debug(f"Changed to: {os.getcwd()}")
                break

    from ai_memory.vector_memory import VectorMemory

    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model BAAI/bge-large-en-v1.5 on %s", device)
        model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
    except Exception:  # ImportError or runtime failure
        logger.warning("sentence_transformers not available, using mock embeddings")
        model = None

    vm = VectorMemory()
    if not vm.load():
        logger.info("Loaded 0 memories")
        print(
            "There is no Claude conversation memory. We didn't import any memories from a Claude conversation. We started our conversation from scratch, and there's no prior conversation or memory to reference. If you'd like to discuss something specific, feel free to let me know, and I'll do my best to help!"
        )
        return 1

    logger.info(f"Loaded {len(vm.memories)} memories")

    results = vm.search(args.query, top_k=10)
    if not results:
        print("No relevant memories found for your query.")
        return 0

    print(f"Found {len(results)} relevant memories from imported conversations:")
    print()
    for i, (entry, score) in enumerate(results[:5], 1):
        ts = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d %H:%M")
        text = entry.text.replace("\n", " ").strip()
        if len(text) > 200:
            text = text[:197] + "..."
        print(f"{i}. [{ts}] (relevance: {score:.3f})")
        print(f"   {text}")
        print()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
