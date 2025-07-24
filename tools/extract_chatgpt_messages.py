#!/usr/bin/env python3
"""Extract and vectorize messages from ChatGPT conversation exports."""
import json
import sys
import subprocess
from pathlib import Path
from typing import List
import tempfile
import os


def extract_messages_from_chatgpt(file_path: Path) -> List[str]:
    """Extract actual message content from ChatGPT export format."""
    with open(file_path) as f:
        data = json.load(f)

    messages: List[str] = []

    for conv in data:
        if "mapping" not in conv:
            continue

        title = conv.get("title", "Untitled")

        # Navigate the mapping structure
        mapping = conv["mapping"]
        for node_id, node in mapping.items():
            if not node.get("message"):
                continue

            msg = node["message"]
            if msg.get("content") and msg["content"].get("parts"):
                parts = msg["content"]["parts"]
                # Join all parts into one text
                text = " ".join(
                    str(part) for part in parts if part and str(part).strip()
                )

                # Only add substantial messages
                if text.strip() and len(text) > 20 and not text.startswith("..."):
                    role = msg.get("author", {}).get("role", "unknown")
                    # Format: [Title] role: message
                    full_text = f"[{title}] {role}: {text}"
                    messages.append(full_text)

    return messages


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: extract_chatgpt_messages.py <conversations.json>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    print(f"Processing {file_path}...")

    messages = extract_messages_from_chatgpt(file_path)
    print(f"Extracted {len(messages)} messages")

    if not messages:
        print("No messages found!")
        return

    # Save as simple JSON array
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(messages, f)
        temp_file = f.name

    # Vectorize using aimem
    vector_dir = Path.home() / "aimemorysystem"
    cmd = [
        "aimem",
        "vectorize",
        temp_file,
        "--json-extract",
        "all",
        "--vector-index",
        str(vector_dir / "memory_store.index"),
        "--model",
        "llama3:70b-instruct-q4_K_M",
    ]

    # Force CPU mode to avoid GPU issues
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    try:
        subprocess.run(cmd, check=True, env=env)
        print(f"Successfully vectorized {len(messages)} messages")
    finally:
        Path(temp_file).unlink()


if __name__ == "__main__":  # pragma: no cover
    main()
