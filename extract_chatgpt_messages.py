#!/usr/bin/env python3
"""Extract and vectorize messages from ChatGPT conversation exports."""

import json
import sys
import subprocess
from pathlib import Path
from typing import List
import tempfile
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_messages_from_chatgpt(file_path: Path) -> List[str]:
    """Extract message text from various ChatGPT export formats."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages: List[str] = []

    # Handle both single conversation and list of conversations
    conversations = data if isinstance(data, list) else [data]

    for conv in conversations:
        # Format 1: mapping structure (newer exports)
        if "mapping" in conv:
            title = conv.get("title", "Untitled")
            mapping = conv["mapping"]

            for node in mapping.values():
                if not node or not isinstance(node, dict):
                    continue

                message = node.get("message")
                if not message or not isinstance(message, dict):
                    continue

                content = message.get("content", {})
                if not isinstance(content, dict):
                    continue

                parts = content.get("parts", [])
                if not parts:
                    continue

                text_parts: List[str] = [p for p in parts if isinstance(p, str) and p.strip()]
                if not text_parts:
                    continue

                text = " ".join(text_parts)
                if len(text) < 20 or text == "...":
                    continue

                author = message.get("author", {})
                role = author.get("role", "unknown") if isinstance(author, dict) else "unknown"
                full_text = f"[{title}] {role}: {text}"
                messages.append(full_text)

        # Format 2: chat_messages structure (older exports)
        elif "chat_messages" in conv:
            name = conv.get("name", "Untitled")
            chat_messages = conv.get("chat_messages", [])

            for msg in chat_messages:
                if not isinstance(msg, dict):
                    continue

                text = msg.get("text", "")

                if not text and "content" in msg:
                    content_list = msg.get("content", [])
                    text_parts = []
                    for item in content_list:
                        if isinstance(item, dict) and "text" in item:
                            text_parts.append(item["text"])
                    text = " ".join(text_parts)

                if text and len(text) > 20:
                    role = msg.get("role", "unknown")
                    if "sender" in msg:
                        sender = msg["sender"]
                        role = sender.get("type", "unknown") if isinstance(sender, dict) else "unknown"

                    full_text = f"[{name}] {role}: {text}"
                    messages.append(full_text)

    return messages

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: extract_chatgpt_messages.py <conversations.json>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        sys.exit(1)

    logger.info("Processing %s...", file_path)

    try:
        messages = extract_messages_from_chatgpt(file_path)
        logger.info("Extracted %d messages", len(messages))

        if not messages:
            logger.warning("No messages found!")
            return

        # Save as temporary JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
            temp_file = f.name

        vector_dir = Path(os.getenv("LUNA_VECTOR_DIR", str(Path.home() / "aimemorysystem")))
        vector_dir.mkdir(parents=True, exist_ok=True)

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

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""

        logger.info("Vectorizing %d messages...", len(messages))
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if result.returncode != 0:
            logger.error("Vectorization failed: %s", result.stderr)
            sys.exit(1)

        logger.info("Successfully vectorized messages")

    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in %s: %s", file_path, e)
        sys.exit(1)
    except Exception as e:  # pragma: no cover - runtime failure
        logger.error("Error processing %s: %s", file_path, e)
        sys.exit(1)
    finally:
        if "temp_file" in locals():
            Path(temp_file).unlink(missing_ok=True)

if __name__ == "__main__":
    main()
