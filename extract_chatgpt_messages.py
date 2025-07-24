#!/usr/bin/env python3
import json
import sys
import subprocess
from pathlib import Path
from typing import List, Dict
import tempfile

def extract_messages_from_chatgpt(file_path: Path) -> List[Dict[str, str]]:
    """Extract actual message content from ChatGPT export format."""
    with open(file_path) as f:
        data = json.load(f)
    
    messages = []
    
    for conv in data:
        if 'mapping' not in conv:
            continue
            
        title = conv.get('title', 'Untitled')
        
        # Extract messages from the mapping
        mapping = conv['mapping']
        for node_id, node in mapping.items():
            if not node.get('message'):
                continue
                
            msg = node['message']
            if msg.get('content') and msg['content'].get('parts'):
                parts = msg['content']['parts']
                # Join all parts into one text
                text = ' '.join(str(part) for part in parts if part and str(part).strip())
                
                if text.strip() and len(text) > 10:  # Only add meaningful messages
                    messages.append({
                        'text': text,
                        'title': title,
                        'role': msg.get('author', {}).get('role', 'unknown'),
                        'timestamp': msg.get('create_time', 0)
                    })
    
    return messages

def main():
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
    
    # Save to temporary file for vectorization
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(messages, f)
        temp_file = f.name
    
    # Vectorize using aimem with CPU
    vector_dir = Path.home() / "aimemorysystem"
    cmd = [
        "aimem", "vectorize", temp_file,
        "--json-extract", "text",
        "--vector-index", str(vector_dir / "memory_store.index"),
        "--model", "llama3:70b-instruct-q4_K_M"
    ]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
    
    subprocess.run(cmd, check=True, env=env)
    Path(temp_file).unlink()  # Clean up temp file

if __name__ == "__main__":
    import os
    main()
