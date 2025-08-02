#!/usr/bin/env python3
"""
chat_with_luna.py – Memory-aware CLI for Luna with token cap enforcement
"""

import argparse
import os
import subprocess
import sys
from ai_memory.vector_memory import VectorMemory
from ai_memory.luna_wrapper import wrap_luna_query

import tiktoken

MAX_TOKENS = 120000  # reserve a little buffer below 4096
ENC = tiktoken.get_encoding("cl100k_base")  # works well for GPT-style LLMs


def count_tokens(text: str) -> int:
    return len(ENC.encode(text))


def trim_to_fit(parts: list[str], max_tokens: int) -> list[str]:
    """
    Accepts a list of strings (context lines) and trims from the front
    to fit within token limits.
    """
    result = []
    total = 0
    for line in reversed(parts):
        tokens = count_tokens(line)
        if total + tokens > max_tokens:
            break
        result.append(line)
        total += tokens
    return list(reversed(result))


def _fetch_memory_context(query: str, model: str) -> list[str]:
    try:
        res = subprocess.run(
            [sys.executable, "-m", "ai_memory.cli", "context", query],
            check=True,
            capture_output=True,
            text=True,
        )
        return res.stdout.strip().splitlines()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"aimem context failed: {exc.stderr}") from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="User prompt to Luna")
    parser.add_argument("--context-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--debug", action="store_true")
    ns = parser.parse_args()

    query = ns.query
    try:
        ctx = _fetch_memory_context(query, ns.context_model)
    except RuntimeError as e:
        print(f"[context error] {e}", file=sys.stderr)
        ctx = []

    vm = VectorMemory(index_path=".ai_memory/memory_store.index")
    memories = vm.search(query, top_k=10)

    # Combine all candidate text for token budgeting
    all_lines = ctx + [f"[MEMORY] {m['text']}" for _, m in memories]
    trimmed_lines = trim_to_fit(all_lines, MAX_TOKENS)

    if len(trimmed_lines) < len(all_lines):
        print(f"[✂] Context trimmed to fit {MAX_TOKENS} tokens", file=sys.stderr)

    response = wrap_luna_query(query, context=trimmed_lines, memory_hits=[])
    print(response)


if __name__ == "__main__":
    raise SystemExit(main())
