#!/usr/bin/env python3
# ~/memory_optimizer/chat_with_luna.py
"""CLI wrapper for sending prompts to Luna, with safe quoting and stdin support."""
#!/bin/bash
# ~/bin/luna – always run memory-backed Luna in the correct venv

source ~/aimemory_cpu/bin/activate
python ~/memory_optimizer/ai_memory/chat_with_luna.py "$@"

import argparse
import os
import subprocess
import sys

OLLAMA_BIN   = os.getenv("OLLAMA_BIN", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "luna")

def clean_prompt(tokens: list[str]) -> str:
    """Join tokens and gently strip matching outer quotes."""
    if not tokens:
        return ""
    prompt = " ".join(tokens)
    if len(prompt) >= 2 and prompt[0] == prompt[-1] and prompt[0] in {"'", '"'}:
        prompt = prompt[1:-1]
    return prompt

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="luna",
        description="Send a prompt to Luna via Ollama",
        add_help=False,
    )
    parser.add_argument("prompt", nargs=argparse.REMAINDER,
                        help="Prompt text (use '-' to read stdin)")
    parser.add_argument("-h", "--help", action="help",
                        help="Show this help and exit")
    args = parser.parse_args()

    if args.prompt == ["-"] or not args.prompt:
        prompt_text = sys.stdin.read().rstrip("\n")
    else:
        prompt_text = clean_prompt(args.prompt)

    if not prompt_text:
        sys.exit("❌  No prompt provided (use '-' to read stdin).")

    result = subprocess.run(
        [OLLAMA_BIN, "run", OLLAMA_MODEL, prompt_text],
        input=b"",  # fix for stdin EOF and stuck '>' prompt
    )
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
