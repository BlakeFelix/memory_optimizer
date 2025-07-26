"""chat_with_luna – memory-augmented chat wrapper for the local 70B model.

Usage (once installed):
    chat-with-luna "Why did we choose FAISS?"

Environment variables respected:
    OLLAMA_MODEL     – model name in Ollama (default: "luna")
    LUNA_VECTOR_DIR  – where the vector index + metadata live
    LUNA_VECTOR_INDEX – override full path to index file
"""
from __future__ import annotations

import os
import subprocess
import sys
import json
import urllib.request
from typing import Optional


# ------------------------------------------------------------------- #
#  Helpers – memory retrieval                                         #
# ------------------------------------------------------------------- #

def _fetch_memory_context(query: str, model_profile: str | None = None) -> str:
    """Call the aimem CLI and return its stdout (stripped)."""
    cmd = [
        sys.executable, "-m", "ai_memory.cli", "context", query
    ]
    if model_profile:
        cmd += ["--model", model_profile]
    # inherit LUNA_VECTOR_… env so aimem sees the same store
    try:
        res = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        return res.stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"aimem context failed: {exc.stderr}") from exc


def _build_prompt(context: str, user_query: str) -> str:
    """Return a single prompt string with context then query."""
    if not context:
        return user_query
    lines: list[str] = ["Relevant context:"]
    for i, snippet in enumerate(context.splitlines(), 1):
        snippet = snippet.strip()
        if snippet:
            lines.append(f"{i}. {snippet}")
    lines.append("")  # blank line
    lines.append(f"User's question: {user_query}")
    lines.append("Answer:")
    return "\n".join(lines)


# ------------------------------------------------------------------- #
#  Helpers – model call (Ollama HTTP API)                             #
# ------------------------------------------------------------------- #

def _query_ollama(model: str, prompt: str) -> str:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode())
    return payload.get("response", "").strip()


# ------------------------------------------------------------------- #
#  Public CLI                                                         #
# ------------------------------------------------------------------- #

def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Ask Luna with automatic memory context"
    )
    ap.add_argument(
        "query",
        help="Your question – enclose in quotes if it contains spaces.",
    )
    ap.add_argument(
        "--context-model", "-c",
        help="Model profile to tell memory_optimizer (e.g. gpt-4-32k, luna).",
    )
    ns = ap.parse_args(argv)

    # Step 1 – gather context
    ctx = _fetch_memory_context(ns.query, ns.context_model)
    # Step 2 – build prompt
    prompt = _build_prompt(ctx, ns.query)
    # Step 3 – send to Luna
    model_name = os.getenv("OLLAMA_MODEL", "luna")
    try:
        answer = _query_ollama(model_name, prompt)
    except Exception as exc:
        print(f"[✗] Failed to reach Ollama: {exc}", file=sys.stderr)
        return 1
    print(answer)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
