#!/usr/bin/env python3
"""Extract and vectorize messages from ChatGPT conversation exports."""
import json
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_messages_from_chatgpt(file_path: Path) -> List[str]:
    """Extract actual message content from ChatGPT export format."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    messages: List[str] = []
    
    # Handle both single conversation and list of conversations
    conversations = data if isinstance(data, list) else [data]

    for conv in conversations:
        if 'mapping' not in conv:
            continue

        title = conv.get('title', 'Untitled')
        mapping = conv['mapping']
        
        for node_id, node in mapping.items():
            if not node or not isinstance(node, dict):
                continue
                
            message = node.get('message')
            if not message or not isinstance(message, dict):
                continue
                
            content = message.get('content', {})
            if not isinstance(content, dict):
                continue
                
            parts = content.get('parts', [])
            if not parts:
                continue

            # Extract text from parts
            text_parts: List[str] = []
            for part in parts:
                if isinstance(part, str) and part.strip():
                    text_parts.append(part)

            if not text_parts:
                continue

            text = ' '.join(text_parts)
            
            # Skip short messages or placeholders
            if len(text) < 20 or text == '...':
                continue

            # Get author role
            author = message.get('author', {})
            role = author.get('role', 'unknown') if isinstance(author, dict) else 'unknown'
            
            # Format message with context
            full_text = f"[{title}] {role}: {text}"
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
            temp_file = f.name

        # Get vector directory from environment or default
        vector_dir = Path(os.getenv('LUNA_VECTOR_DIR', str(Path.home() / 'aimemorysystem')))
        vector_dir.mkdir(parents=True, exist_ok=True)

        # Build vectorization command
        cmd = [
            'aimem', 'vectorize', temp_file,
            '--json-extract', 'all',  # Use 'all' since we pre-extracted strings
            '--vector-index', str(vector_dir / 'memory_store.index'),
            '--model', 'llama3:70b-instruct-q4_K_M'
        ]

        # Force CPU mode
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ''

        logger.info('Vectorizing %d messages...', len(messages))
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            logger.error('Vectorization failed: %s', result.stderr)
            sys.exit(1)

        logger.info('Successfully vectorized messages')
        
    except json.JSONDecodeError as e:
        logger.error('Invalid JSON in %s: %s', file_path, e)
        sys.exit(1)
    except Exception as e:
        logger.error('Error processing %s: %s', file_path, e)
        sys.exit(1)
    finally:
        if 'temp_file' in locals():
            Path(temp_file).unlink(missing_ok=True)


if __name__ == '__main__':
    main()